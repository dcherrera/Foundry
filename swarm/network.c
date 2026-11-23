/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "network.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

/* ============================================================= */
/* Internal Structures                                           */
/* ============================================================= */

struct sw_socket {
    int fd;
    sw_conn_state state;
    char remote_addr[64];
    uint16_t remote_port;
    char local_addr[64];
    uint16_t local_port;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t last_activity;
    bool is_server;
};

struct sw_socket_set {
    fd_set read_fds;
    fd_set write_fds;
    fd_set error_fds;
    fd_set result_read;
    fd_set result_write;
    fd_set result_error;
    int max_fd;
    sw_socket** sockets;
    int num_sockets;
    int capacity;
};

/* Thread-local error message */
static __thread char last_error[256] = {0};

static void set_error(const char* msg) {
    snprintf(last_error, sizeof(last_error), "%s: %s", msg, strerror(errno));
}

static uint64_t current_time(void) {
    return (uint64_t)time(NULL);
}

/* ============================================================= */
/* Server Operations                                             */
/* ============================================================= */

sw_socket* sw_server_create(uint16_t port) {
    sw_socket* sock = calloc(1, sizeof(sw_socket));
    if (!sock) {
        set_error("Failed to allocate socket");
        return NULL;
    }

    sock->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock->fd < 0) {
        set_error("Failed to create socket");
        free(sock);
        return NULL;
    }

    /* Allow address reuse */
    int opt = 1;
    setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock->fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        set_error("Failed to bind socket");
        close(sock->fd);
        free(sock);
        return NULL;
    }

    if (listen(sock->fd, SW_MAX_PENDING) < 0) {
        set_error("Failed to listen on socket");
        close(sock->fd);
        free(sock);
        return NULL;
    }

    sock->is_server = true;
    sock->state = SW_CONN_CONNECTED;
    sock->local_port = port;
    snprintf(sock->local_addr, sizeof(sock->local_addr), "0.0.0.0");
    sock->last_activity = current_time();

    return sock;
}

sw_socket* sw_server_accept(sw_socket* server, int timeout_sec) {
    if (!server || !server->is_server) {
        set_error("Invalid server socket");
        return NULL;
    }

    /* Wait for connection with timeout */
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(server->fd, &fds);

    struct timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;

    int ret = select(server->fd + 1, &fds, NULL, NULL, &tv);
    if (ret < 0) {
        set_error("Select failed");
        return NULL;
    }
    if (ret == 0) {
        /* Timeout - not an error */
        return NULL;
    }

    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int client_fd = accept(server->fd, (struct sockaddr*)&client_addr, &addr_len);
    if (client_fd < 0) {
        set_error("Accept failed");
        return NULL;
    }

    sw_socket* client = calloc(1, sizeof(sw_socket));
    if (!client) {
        close(client_fd);
        set_error("Failed to allocate client socket");
        return NULL;
    }

    client->fd = client_fd;
    client->state = SW_CONN_CONNECTED;
    client->remote_port = ntohs(client_addr.sin_port);
    inet_ntop(AF_INET, &client_addr.sin_addr, client->remote_addr, sizeof(client->remote_addr));
    client->last_activity = current_time();

    /* Enable TCP_NODELAY for lower latency */
    int flag = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    server->last_activity = current_time();

    return client;
}

sw_socket* sw_server_accept_nonblock(sw_socket* server) {
    if (!server || !server->is_server) return NULL;

    /* Set non-blocking temporarily */
    int flags = fcntl(server->fd, F_GETFL, 0);
    fcntl(server->fd, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int client_fd = accept(server->fd, (struct sockaddr*)&client_addr, &addr_len);

    /* Restore blocking mode */
    fcntl(server->fd, F_SETFL, flags);

    if (client_fd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            /* No pending connection - not an error */
            return NULL;
        }
        set_error("Accept failed");
        return NULL;
    }

    sw_socket* client = calloc(1, sizeof(sw_socket));
    if (!client) {
        close(client_fd);
        return NULL;
    }

    client->fd = client_fd;
    client->state = SW_CONN_CONNECTED;
    client->remote_port = ntohs(client_addr.sin_port);
    inet_ntop(AF_INET, &client_addr.sin_addr, client->remote_addr, sizeof(client->remote_addr));
    client->last_activity = current_time();

    int flag = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    return client;
}

/* ============================================================= */
/* Client Operations                                             */
/* ============================================================= */

sw_socket* sw_client_connect(const char* host, uint16_t port, int timeout_sec) {
    if (!host) {
        set_error("Invalid host");
        return NULL;
    }

    sw_socket* sock = calloc(1, sizeof(sw_socket));
    if (!sock) {
        set_error("Failed to allocate socket");
        return NULL;
    }

    sock->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock->fd < 0) {
        set_error("Failed to create socket");
        free(sock);
        return NULL;
    }

    /* Resolve hostname */
    struct hostent* he = gethostbyname(host);
    if (!he) {
        set_error("Failed to resolve hostname");
        close(sock->fd);
        free(sock);
        return NULL;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    addr.sin_port = htons(port);

    inet_ntop(AF_INET, &addr.sin_addr, sock->remote_addr, sizeof(sock->remote_addr));
    sock->remote_port = port;
    sock->state = SW_CONN_CONNECTING;

    /* Set non-blocking for connect with timeout */
    int flags = fcntl(sock->fd, F_GETFL, 0);
    fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK);

    int ret = connect(sock->fd, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0 && errno != EINPROGRESS) {
        set_error("Connect failed");
        close(sock->fd);
        free(sock);
        return NULL;
    }

    /* Wait for connection with timeout */
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock->fd, &fds);

    struct timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;

    ret = select(sock->fd + 1, NULL, &fds, NULL, &tv);
    if (ret <= 0) {
        set_error(ret == 0 ? "Connection timed out" : "Select failed");
        close(sock->fd);
        free(sock);
        return NULL;
    }

    /* Check for connection error */
    int error = 0;
    socklen_t len = sizeof(error);
    getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, &error, &len);
    if (error != 0) {
        errno = error;
        set_error("Connection failed");
        close(sock->fd);
        free(sock);
        return NULL;
    }

    /* Restore blocking mode */
    fcntl(sock->fd, F_SETFL, flags);

    sock->state = SW_CONN_CONNECTED;
    sock->last_activity = current_time();

    /* Enable TCP_NODELAY */
    int flag = 1;
    setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    return sock;
}

bool sw_client_reconnect(sw_socket* sock, int timeout_sec) {
    if (!sock || sock->is_server) return false;

    if (sock->fd >= 0) {
        close(sock->fd);
    }

    sock->fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock->fd < 0) {
        set_error("Failed to create socket");
        sock->state = SW_CONN_ERROR;
        return false;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    inet_pton(AF_INET, sock->remote_addr, &addr.sin_addr);
    addr.sin_port = htons(sock->remote_port);

    sock->state = SW_CONN_CONNECTING;

    /* Same connection logic as sw_client_connect */
    int flags = fcntl(sock->fd, F_GETFL, 0);
    fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK);

    int ret = connect(sock->fd, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0 && errno != EINPROGRESS) {
        set_error("Reconnect failed");
        sock->state = SW_CONN_ERROR;
        return false;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock->fd, &fds);

    struct timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;

    ret = select(sock->fd + 1, NULL, &fds, NULL, &tv);
    if (ret <= 0) {
        set_error(ret == 0 ? "Reconnection timed out" : "Select failed");
        sock->state = SW_CONN_ERROR;
        return false;
    }

    int error = 0;
    socklen_t len = sizeof(error);
    getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, &error, &len);
    if (error != 0) {
        errno = error;
        set_error("Reconnection failed");
        sock->state = SW_CONN_ERROR;
        return false;
    }

    fcntl(sock->fd, F_SETFL, flags);

    sock->state = SW_CONN_CONNECTED;
    sock->last_activity = current_time();

    int flag = 1;
    setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    return true;
}

/* ============================================================= */
/* Common Operations                                             */
/* ============================================================= */

void sw_socket_close(sw_socket* sock) {
    if (!sock) return;
    if (sock->fd >= 0) {
        close(sock->fd);
    }
    free(sock);
}

sw_socket_info sw_socket_get_info(sw_socket* sock) {
    sw_socket_info info = {0};
    if (!sock) return info;

    info.fd = sock->fd;
    info.state = sock->state;
    strncpy(info.remote_addr, sock->remote_addr, sizeof(info.remote_addr) - 1);
    info.remote_port = sock->remote_port;
    info.bytes_sent = sock->bytes_sent;
    info.bytes_received = sock->bytes_received;
    info.last_activity = sock->last_activity;

    return info;
}

bool sw_socket_is_connected(sw_socket* sock) {
    return sock && sock->state == SW_CONN_CONNECTED;
}

int sw_socket_fd(sw_socket* sock) {
    return sock ? sock->fd : -1;
}

/* ============================================================= */
/* Message I/O                                                   */
/* ============================================================= */

bool sw_send_msg(sw_socket* sock, uint8_t* msg) {
    if (!sock || !msg || sock->state != SW_CONN_CONNECTED) {
        return false;
    }

    uint32_t total = sw_msg_total_size(msg);
    size_t sent = 0;

    while (sent < total) {
        ssize_t n = send(sock->fd, msg + sent, total - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* Would block - wait and retry */
                fd_set fds;
                FD_ZERO(&fds);
                FD_SET(sock->fd, &fds);
                struct timeval tv = {SW_IO_TIMEOUT, 0};
                if (select(sock->fd + 1, NULL, &fds, NULL, &tv) <= 0) {
                    set_error("Send timeout");
                    sock->state = SW_CONN_ERROR;
                    return false;
                }
                continue;
            }
            set_error("Send failed");
            sock->state = SW_CONN_ERROR;
            return false;
        }
        if (n == 0) {
            set_error("Connection closed");
            sock->state = SW_CONN_DISCONNECTED;
            return false;
        }
        sent += n;
    }

    sock->bytes_sent += total;
    sock->last_activity = current_time();
    return true;
}

int sw_send_bytes(sw_socket* sock, const void* data, size_t len) {
    if (!sock || !data || sock->state != SW_CONN_CONNECTED) {
        return -1;
    }

    ssize_t n = send(sock->fd, data, len, MSG_NOSIGNAL);
    if (n > 0) {
        sock->bytes_sent += n;
        sock->last_activity = current_time();
    } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        set_error("Send failed");
        sock->state = SW_CONN_ERROR;
    }

    return (int)n;
}

uint8_t* sw_recv_msg(sw_socket* sock, int timeout_sec) {
    if (!sock || sock->state != SW_CONN_CONNECTED) {
        return NULL;
    }

    /* First, receive header */
    uint8_t header_buf[SW_HEADER_SIZE];
    size_t received = 0;

    time_t start = time(NULL);

    while (received < SW_HEADER_SIZE) {
        /* Check timeout */
        if (time(NULL) - start > timeout_sec) {
            set_error("Receive timeout");
            return NULL;
        }

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(sock->fd, &fds);
        struct timeval tv = {1, 0};  /* 1 second intervals */

        int ret = select(sock->fd + 1, &fds, NULL, NULL, &tv);
        if (ret < 0) {
            if (errno == EINTR) continue;
            set_error("Select failed");
            sock->state = SW_CONN_ERROR;
            return NULL;
        }
        if (ret == 0) continue;  /* Timeout, loop again */

        ssize_t n = recv(sock->fd, header_buf + received, SW_HEADER_SIZE - received, 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            set_error("Receive failed");
            sock->state = SW_CONN_ERROR;
            return NULL;
        }
        if (n == 0) {
            set_error("Connection closed");
            sock->state = SW_CONN_DISCONNECTED;
            return NULL;
        }
        received += n;
    }

    /* Validate header */
    sw_msg_header* hdr = (sw_msg_header*)header_buf;
    if (hdr->magic != SW_PROTOCOL_MAGIC) {
        set_error("Invalid message magic");
        return NULL;
    }
    if (hdr->length > SW_MAX_MESSAGE_SIZE - SW_HEADER_SIZE) {
        set_error("Message too large");
        return NULL;
    }

    /* Allocate full message buffer */
    uint32_t total = SW_HEADER_SIZE + hdr->length;
    uint8_t* msg = malloc(total);
    if (!msg) {
        set_error("Failed to allocate message buffer");
        return NULL;
    }

    memcpy(msg, header_buf, SW_HEADER_SIZE);

    /* Receive payload */
    received = 0;
    while (received < hdr->length) {
        if (time(NULL) - start > timeout_sec) {
            set_error("Receive timeout");
            free(msg);
            return NULL;
        }

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(sock->fd, &fds);
        struct timeval tv = {1, 0};

        int ret = select(sock->fd + 1, &fds, NULL, NULL, &tv);
        if (ret < 0) {
            if (errno == EINTR) continue;
            set_error("Select failed");
            sock->state = SW_CONN_ERROR;
            free(msg);
            return NULL;
        }
        if (ret == 0) continue;

        ssize_t n = recv(sock->fd, msg + SW_HEADER_SIZE + received,
                         hdr->length - received, 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            set_error("Receive failed");
            sock->state = SW_CONN_ERROR;
            free(msg);
            return NULL;
        }
        if (n == 0) {
            set_error("Connection closed");
            sock->state = SW_CONN_DISCONNECTED;
            free(msg);
            return NULL;
        }
        received += n;
    }

    sock->bytes_received += total;
    sock->last_activity = current_time();

    return msg;
}

int sw_recv_bytes(sw_socket* sock, void* buf, size_t len, int timeout_sec) {
    if (!sock || !buf || sock->state != SW_CONN_CONNECTED) {
        return -1;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock->fd, &fds);
    struct timeval tv = {timeout_sec, 0};

    int ret = select(sock->fd + 1, &fds, NULL, NULL, &tv);
    if (ret < 0) {
        set_error("Select failed");
        return -1;
    }
    if (ret == 0) {
        return 0;  /* Timeout */
    }

    ssize_t n = recv(sock->fd, buf, len, 0);
    if (n > 0) {
        sock->bytes_received += n;
        sock->last_activity = current_time();
    } else if (n == 0) {
        sock->state = SW_CONN_DISCONNECTED;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        set_error("Receive failed");
        sock->state = SW_CONN_ERROR;
    }

    return (int)n;
}

bool sw_data_available(sw_socket* sock) {
    if (!sock || sock->fd < 0) return false;

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(sock->fd, &fds);
    struct timeval tv = {0, 0};  /* Non-blocking check */

    return select(sock->fd + 1, &fds, NULL, NULL, &tv) > 0;
}

/* ============================================================= */
/* Multiplexing                                                  */
/* ============================================================= */

sw_socket_set* sw_socket_set_new(void) {
    sw_socket_set* set = calloc(1, sizeof(sw_socket_set));
    if (!set) return NULL;

    set->capacity = 16;
    set->sockets = calloc(set->capacity, sizeof(sw_socket*));
    if (!set->sockets) {
        free(set);
        return NULL;
    }

    FD_ZERO(&set->read_fds);
    FD_ZERO(&set->write_fds);
    FD_ZERO(&set->error_fds);
    set->max_fd = -1;

    return set;
}

void sw_socket_set_free(sw_socket_set* set) {
    if (!set) return;
    free(set->sockets);
    free(set);
}

void sw_socket_set_add(sw_socket_set* set, sw_socket* sock) {
    if (!set || !sock || sock->fd < 0) return;

    /* Expand if needed */
    if (set->num_sockets >= set->capacity) {
        int new_cap = set->capacity * 2;
        sw_socket** new_sockets = realloc(set->sockets, new_cap * sizeof(sw_socket*));
        if (!new_sockets) return;
        set->sockets = new_sockets;
        set->capacity = new_cap;
    }

    set->sockets[set->num_sockets++] = sock;
    FD_SET(sock->fd, &set->read_fds);
    FD_SET(sock->fd, &set->error_fds);

    if (sock->fd > set->max_fd) {
        set->max_fd = sock->fd;
    }
}

void sw_socket_set_remove(sw_socket_set* set, sw_socket* sock) {
    if (!set || !sock) return;

    for (int i = 0; i < set->num_sockets; i++) {
        if (set->sockets[i] == sock) {
            /* Remove from array */
            memmove(&set->sockets[i], &set->sockets[i + 1],
                    (set->num_sockets - i - 1) * sizeof(sw_socket*));
            set->num_sockets--;

            /* Remove from fd_sets */
            if (sock->fd >= 0) {
                FD_CLR(sock->fd, &set->read_fds);
                FD_CLR(sock->fd, &set->write_fds);
                FD_CLR(sock->fd, &set->error_fds);
            }

            /* Recalculate max_fd */
            set->max_fd = -1;
            for (int j = 0; j < set->num_sockets; j++) {
                if (set->sockets[j]->fd > set->max_fd) {
                    set->max_fd = set->sockets[j]->fd;
                }
            }
            break;
        }
    }
}

int sw_socket_set_wait(sw_socket_set* set, int timeout_ms) {
    if (!set || set->num_sockets == 0) return 0;

    /* Copy fd_sets since select modifies them */
    set->result_read = set->read_fds;
    set->result_write = set->write_fds;
    set->result_error = set->error_fds;

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    return select(set->max_fd + 1, &set->result_read, &set->result_write,
                  &set->result_error, &tv);
}

bool sw_socket_set_is_readable(sw_socket_set* set, sw_socket* sock) {
    if (!set || !sock || sock->fd < 0) return false;
    return FD_ISSET(sock->fd, &set->result_read);
}

bool sw_socket_set_is_writable(sw_socket_set* set, sw_socket* sock) {
    if (!set || !sock || sock->fd < 0) return false;
    return FD_ISSET(sock->fd, &set->result_write);
}

bool sw_socket_set_has_error(sw_socket_set* set, sw_socket* sock) {
    if (!set || !sock || sock->fd < 0) return false;
    return FD_ISSET(sock->fd, &set->result_error);
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

const char* sw_socket_error(void) {
    return last_error;
}

bool sw_resolve_host(const char* hostname, char* ip_out, size_t ip_len) {
    if (!hostname || !ip_out) return false;

    struct hostent* he = gethostbyname(hostname);
    if (!he) return false;

    inet_ntop(AF_INET, he->h_addr_list[0], ip_out, ip_len);
    return true;
}

bool sw_get_local_ip(char* ip_out, size_t ip_len) {
    if (!ip_out) return false;

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) < 0) {
        return false;
    }

    return sw_resolve_host(hostname, ip_out, ip_len);
}

bool sw_set_nonblocking(sw_socket* sock, bool nonblocking) {
    if (!sock || sock->fd < 0) return false;

    int flags = fcntl(sock->fd, F_GETFL, 0);
    if (flags < 0) return false;

    if (nonblocking) {
        flags |= O_NONBLOCK;
    } else {
        flags &= ~O_NONBLOCK;
    }

    return fcntl(sock->fd, F_SETFL, flags) >= 0;
}

bool sw_set_nodelay(sw_socket* sock, bool nodelay) {
    if (!sock || sock->fd < 0) return false;
    int flag = nodelay ? 1 : 0;
    return setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) >= 0;
}

bool sw_set_keepalive(sw_socket* sock, bool keepalive) {
    if (!sock || sock->fd < 0) return false;
    int flag = keepalive ? 1 : 0;
    return setsockopt(sock->fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag)) >= 0;
}
