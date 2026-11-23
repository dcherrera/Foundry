/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_SWARM_NETWORK_H
#define FOUNDRY_SWARM_NETWORK_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Swarm Network - TCP socket communication layer.
 *
 * Provides:
 * - Server socket for coordinator
 * - Client socket for workers
 * - Non-blocking I/O with select/poll
 * - Message framing (handles partial reads/writes)
 */

/* Default port */
#define SW_DEFAULT_PORT 7100

/* Connection timeout (seconds) */
#define SW_CONNECT_TIMEOUT 30

/* Read/write timeout (seconds) */
#define SW_IO_TIMEOUT 60

/* Maximum pending connections */
#define SW_MAX_PENDING 64

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Opaque socket handle */
typedef struct sw_socket sw_socket;

/* Connection state */
typedef enum {
    SW_CONN_DISCONNECTED,
    SW_CONN_CONNECTING,
    SW_CONN_CONNECTED,
    SW_CONN_ERROR,
} sw_conn_state;

/* Socket info */
typedef struct {
    int fd;
    sw_conn_state state;
    char remote_addr[64];
    uint16_t remote_port;
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t last_activity;     /* Unix timestamp */
} sw_socket_info;

/* ============================================================= */
/* Server Operations                                             */
/* ============================================================= */

/* Create a listening server socket */
sw_socket* sw_server_create(uint16_t port);

/* Accept a new connection (blocks until connection or timeout) */
sw_socket* sw_server_accept(sw_socket* server, int timeout_sec);

/* Accept a new connection (non-blocking, returns NULL if no pending) */
sw_socket* sw_server_accept_nonblock(sw_socket* server);

/* ============================================================= */
/* Client Operations                                             */
/* ============================================================= */

/* Connect to a server */
sw_socket* sw_client_connect(const char* host, uint16_t port, int timeout_sec);

/* Reconnect after disconnect */
bool sw_client_reconnect(sw_socket* sock, int timeout_sec);

/* ============================================================= */
/* Common Operations                                             */
/* ============================================================= */

/* Close and free socket */
void sw_socket_close(sw_socket* sock);

/* Get socket info */
sw_socket_info sw_socket_get_info(sw_socket* sock);

/* Check if socket is connected */
bool sw_socket_is_connected(sw_socket* sock);

/* Get underlying file descriptor (for select/poll) */
int sw_socket_fd(sw_socket* sock);

/* ============================================================= */
/* Message I/O                                                   */
/* ============================================================= */

/* Send a complete message (handles partial writes)
 * Returns: true on success, false on error
 */
bool sw_send_msg(sw_socket* sock, uint8_t* msg);

/* Send raw bytes
 * Returns: bytes sent, or -1 on error
 */
int sw_send_bytes(sw_socket* sock, const void* data, size_t len);

/* Receive a complete message (handles partial reads)
 * Allocates message buffer, caller must free with sw_msg_free()
 * Returns: message buffer, or NULL on error/timeout
 */
uint8_t* sw_recv_msg(sw_socket* sock, int timeout_sec);

/* Receive raw bytes
 * Returns: bytes received, 0 on disconnect, -1 on error
 */
int sw_recv_bytes(sw_socket* sock, void* buf, size_t len, int timeout_sec);

/* Check if data is available to read (non-blocking) */
bool sw_data_available(sw_socket* sock);

/* ============================================================= */
/* Multiplexing                                                  */
/* ============================================================= */

/* Socket set for select-style multiplexing */
typedef struct sw_socket_set sw_socket_set;

/* Create a socket set */
sw_socket_set* sw_socket_set_new(void);

/* Free a socket set */
void sw_socket_set_free(sw_socket_set* set);

/* Add socket to set */
void sw_socket_set_add(sw_socket_set* set, sw_socket* sock);

/* Remove socket from set */
void sw_socket_set_remove(sw_socket_set* set, sw_socket* sock);

/* Wait for activity on any socket in set
 * Returns: number of ready sockets, or -1 on error
 */
int sw_socket_set_wait(sw_socket_set* set, int timeout_ms);

/* Check if socket is ready for reading after wait */
bool sw_socket_set_is_readable(sw_socket_set* set, sw_socket* sock);

/* Check if socket is ready for writing after wait */
bool sw_socket_set_is_writable(sw_socket_set* set, sw_socket* sock);

/* Check if socket has error after wait */
bool sw_socket_set_has_error(sw_socket_set* set, sw_socket* sock);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Get last error message */
const char* sw_socket_error(void);

/* Resolve hostname to IP address */
bool sw_resolve_host(const char* hostname, char* ip_out, size_t ip_len);

/* Get local IP address */
bool sw_get_local_ip(char* ip_out, size_t ip_len);

/* Set socket to non-blocking mode */
bool sw_set_nonblocking(sw_socket* sock, bool nonblocking);

/* Set TCP_NODELAY (disable Nagle's algorithm) */
bool sw_set_nodelay(sw_socket* sock, bool nodelay);

/* Set SO_KEEPALIVE */
bool sw_set_keepalive(sw_socket* sock, bool keepalive);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_SWARM_NETWORK_H */
