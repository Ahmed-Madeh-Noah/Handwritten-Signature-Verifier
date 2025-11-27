# Network Architecture & Port Forwarding

This tool is designed to be self-hosted and served at a custom domain or subdomain.

### The Request Flow
Technically, this happens in two distinct phases:

1.  **DNS Resolution:** Client &rarr; Domain Provider (DNS) &rarr; Returns Server's Router Public IP.
2.  **Data Transfer:** Client &rarr; Server's Router &rarr; Server Device &rarr; Reverse Proxy &rarr; Flask &rarr; Model.

### Execution Details

-   **Client Request &rarr; Server's Router IP (DDNS):**
    The client performs a DNS lookup to find the router. Since residential router IPs are often dynamic (changing), a **DDNS (Dynamic DNS)** tool is set up on the server. This tool periodically checks the public IP and uses an API call to the domain provider to update the A-Record if the IP has changed.

-   **Server's Router IP &rarr; Server Device Local IP (Port Forwarding):**
    The router receives the request but needs to know which device handles it.
    1.  **DHCP Reservation:** The server device is assigned a static local IP (e.g., `192.168.1.218`) via the router's portal (often `192.168.1.1`) using the device's MAC address.
    2.  **Port Forwarding:** Rules are created on the router to forward external ports **80 (HTTP)** and **443 (HTTPS)** to the server device's local IP on the same ports.

-   **Server Device &rarr; Reverse Proxy &rarr; Flask &rarr; Model:**
    This is orchestrated via **Docker Compose**.
    * **Reverse Proxy with Caddy:** Listens on ports 80/443, terminates SSL (if configured), and routes traffic to the Flask container.
    * **Internal Communication:** The containers communicate over a shared Docker network using service names (e.g., `http://flask:5000`) rather than exposing internal logic ports to the outside world.
