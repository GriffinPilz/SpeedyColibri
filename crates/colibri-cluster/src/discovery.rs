//! Discover other DGX Sparks on the ConnectX/RoCE fabric, for a human-verifiable
//! "who is on the wire" report at startup.
//!
//! Two signals are combined:
//!   1. **Local ConnectX links** — the RoCE devices under `/sys/class/infiniband`,
//!      each mapped to its netdev, MAC, IPv4, and link state. These are the ports
//!      that carry the 200 GbE Spark-to-Spark traffic.
//!   2. **Peers on those links** — found two ways and merged by IP:
//!      - a small UDP **beacon** each colibrì node broadcasts on its RoCE subnet,
//!        so nodes already running the engine announce their rank + serve port;
//!      - the kernel **ARP/neighbor table** (primed by a quick subnet sweep),
//!        classified as a likely Spark when the neighbor's MAC OUI matches one of
//!        our own ConnectX NICs.
//!
//! Pure `std` (Linux): `/sys` + `/proc` reads, `UdpSocket`, and the `ip` tool for
//! interface addresses. On non-Linux it degrades to an empty result — the fabric
//! only exists on the Spark.

use std::collections::HashMap;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// UDP port the discovery beacon broadcasts and listens on.
pub const DISC_PORT: u16 = 48757;
/// Beacon wire prefix + version.
const BEACON_MAGIC: &str = "COLISPRK1";
/// Only sweep subnets no larger than this many hosts (a /24 is 254).
const MAX_SWEEP_HOSTS: u32 = 512;

/// A local ConnectX/RoCE link.
#[derive(Debug, Clone)]
pub struct ConnectXLink {
    /// RDMA device name, e.g. `rocep1s0f0`.
    pub rdma_dev: String,
    /// Backing netdev, e.g. `enp1s0f0np0`.
    pub netdev: String,
    /// MAC address of the netdev, e.g. `30:c5:99:40:42:b3`.
    pub mac: String,
    /// IPv4 assigned to the netdev, if any.
    pub ipv4: Option<Ipv4Addr>,
    /// CIDR prefix length of that address.
    pub prefix: u8,
    /// Port state is `ACTIVE` and the physical link is up.
    pub active: bool,
}

impl ConnectXLink {
    /// First three MAC octets (OUI), lowercased, e.g. `30:c5:99`.
    fn oui(&self) -> String {
        oui_of(&self.mac)
    }
}

/// How a peer was identified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerKind {
    /// Announced itself via the colibrì beacon (running the engine).
    Coli,
    /// Only seen as an L2/L3 neighbor on the fabric (not [yet] running colibrì).
    FabricNeighbor,
}

/// A discovered node on the fabric.
#[derive(Debug, Clone)]
pub struct Peer {
    pub ip: Ipv4Addr,
    pub mac: Option<String>,
    pub hostname: Option<String>,
    pub rank: Option<u32>,
    pub serve_port: Option<u16>,
    /// Local netdev the peer was seen through.
    pub via: String,
    pub kind: PeerKind,
    /// MAC OUI matches one of our local ConnectX NICs (i.e. same hardware family).
    pub nic_match: bool,
}

/// Result of a discovery sweep.
#[derive(Debug, Clone)]
pub struct Discovery {
    pub hostname: String,
    pub links: Vec<ConnectXLink>,
    pub peers: Vec<Peer>,
}

impl Discovery {
    /// Active links (up, with an IPv4) — the ones that can actually carry traffic.
    pub fn active_links(&self) -> impl Iterator<Item = &ConnectXLink> {
        self.links.iter().filter(|l| l.active && l.ipv4.is_some())
    }
    /// Peers running colibrì (announced via beacon).
    pub fn coli_peers(&self) -> impl Iterator<Item = &Peer> {
        self.peers.iter().filter(|p| p.kind == PeerKind::Coli)
    }
}

// ---- local identity --------------------------------------------------------

/// This node's hostname (`/proc/sys/kernel/hostname`, else `$HOSTNAME`, else `?`).
pub fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "?".to_string())
}

fn oui_of(mac: &str) -> String {
    mac.split(':').take(3).collect::<Vec<_>>().join(":").to_ascii_lowercase()
}

// ---- ConnectX link enumeration --------------------------------------------

/// Enumerate the local ConnectX/RoCE links from `/sys/class/infiniband`.
pub fn connectx_links() -> Vec<ConnectXLink> {
    let ipmap = read_ip_addrs();
    let mut links = Vec::new();
    let dir = match std::fs::read_dir("/sys/class/infiniband") {
        Ok(d) => d,
        Err(_) => return links, // not a Linux/RDMA host
    };
    for ent in dir.flatten() {
        let rdma_dev = ent.file_name().to_string_lossy().into_owned();
        let base = ent.path();
        // netdev backing this RDMA device (usually one under device/net/).
        let netdev = std::fs::read_dir(base.join("device/net"))
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned());
        let netdev = match netdev {
            Some(n) => n,
            None => continue,
        };
        let state = read_trim(base.join("ports/1/state")); // "4: ACTIVE"
        let phys = read_trim(base.join("ports/1/phys_state")); // "5: LinkUp"
        let active = state.contains("ACTIVE") && phys.to_ascii_lowercase().contains("linkup");
        let mac = read_trim(format!("/sys/class/net/{netdev}/address"));
        let (ipv4, prefix) = match ipmap.get(&netdev) {
            Some(&(ip, pfx, _)) => (Some(ip), pfx),
            None => (None, 0),
        };
        links.push(ConnectXLink { rdma_dev, netdev, mac, ipv4, prefix, active });
    }
    links.sort_by(|a, b| a.netdev.cmp(&b.netdev));
    links
}

fn read_trim(path: impl AsRef<std::path::Path>) -> String {
    std::fs::read_to_string(path).map(|s| s.trim().to_string()).unwrap_or_default()
}

/// Parse `ip -o -4 addr show` into `netdev -> (ip, prefix, broadcast)`.
fn read_ip_addrs() -> HashMap<String, (Ipv4Addr, u8, Option<Ipv4Addr>)> {
    let mut map = HashMap::new();
    let out = match std::process::Command::new("ip").args(["-o", "-4", "addr", "show"]).output() {
        Ok(o) => o,
        Err(_) => return map,
    };
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        // e.g. "2: enp1s0f0np0 inet 192.168.100.11/24 brd 192.168.100.255 ..."
        let iface = match t.get(1) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let inet = t.iter().position(|&x| x == "inet");
        let cidr = inet.and_then(|i| t.get(i + 1));
        if let Some(cidr) = cidr {
            if let Some((ip, pfx)) = parse_cidr(cidr) {
                let brd = t
                    .iter()
                    .position(|&x| x == "brd")
                    .and_then(|i| t.get(i + 1))
                    .and_then(|s| s.parse::<Ipv4Addr>().ok());
                map.insert(iface, (ip, pfx, brd));
            }
        }
    }
    map
}

fn parse_cidr(s: &str) -> Option<(Ipv4Addr, u8)> {
    let (ip, pfx) = s.split_once('/')?;
    Some((ip.parse().ok()?, pfx.parse().ok()?))
}

/// Broadcast address for `ip/prefix` (e.g. 192.168.100.11/24 -> 192.168.100.255).
pub fn broadcast_addr(ip: Ipv4Addr, prefix: u8) -> Ipv4Addr {
    if prefix == 0 {
        return Ipv4Addr::BROADCAST;
    }
    let ipn = u32::from(ip);
    let mask = if prefix >= 32 { u32::MAX } else { !(u32::MAX >> prefix) };
    Ipv4Addr::from(ipn | !mask)
}

// ---- ARP / neighbor table --------------------------------------------------

/// Read `/proc/net/arp`: `(ip, mac, device)` for each complete entry.
fn read_arp() -> Vec<(Ipv4Addr, String, String)> {
    let text = match std::fs::read_to_string("/proc/net/arp") {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        // IP address  HW type  Flags  HW address  Mask  Device
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 6 {
            continue;
        }
        let flags = f[2];
        let mac = f[3];
        // 0x0 = incomplete; skip those and the null MAC.
        if flags == "0x0" || mac == "00:00:00:00:00:00" {
            continue;
        }
        if let Ok(ip) = f[0].parse::<Ipv4Addr>() {
            out.push((ip, mac.to_ascii_lowercase(), f[5].to_string()));
        }
    }
    out
}

/// Prime the kernel ARP table by firing a 1-byte UDP datagram at every host in
/// each active link's subnet (fire-and-forget; the kernel resolves the MAC).
fn sweep_subnets(links: &[ConnectXLink]) {
    let sock = match UdpSocket::bind("0.0.0.0:0") {
        Ok(s) => s,
        Err(_) => return,
    };
    let _ = sock.set_broadcast(true);
    for l in links.iter().filter(|l| l.active) {
        let (ip, pfx) = match (l.ipv4, l.prefix) {
            (Some(ip), pfx) if pfx >= 8 => (ip, pfx),
            _ => continue,
        };
        let hosts = 1u32 << (32 - pfx as u32);
        if hosts.saturating_sub(2) > MAX_SWEEP_HOSTS {
            continue; // subnet too large to sweep politely
        }
        let base = u32::from(ip) & (u32::MAX << (32 - pfx as u32));
        let selfn = u32::from(ip);
        for h in 1..hosts.saturating_sub(1) {
            let addr = base | h;
            if addr == selfn {
                continue;
            }
            let _ = sock.send_to(&[0u8], (Ipv4Addr::from(addr), 9));
        }
    }
}

// ---- beacon ----------------------------------------------------------------

/// Per-process nonce so a node ignores its own broadcast.
fn make_nonce() -> u64 {
    // Read exactly 8 bytes — `/dev/urandom` is an infinite stream, so a full-file
    // read (fs::read) never returns.
    use std::io::Read;
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let mut b = [0u8; 8];
        if f.read_exact(&mut b).is_ok() {
            return u64::from_le_bytes(b);
        }
    }
    // Fallback: time XOR pid (only needs to be locally unique per process).
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    t ^ ((std::process::id() as u64) << 32)
}

fn beacon_msg(nonce: u64, rank: u32, serve_port: u16, host: &str) -> String {
    // Keep the hostname field free of '|'.
    let host = host.replace('|', "_");
    format!("{BEACON_MAGIC}|{nonce:016x}|{rank}|{serve_port}|{host}")
}

/// Parse a beacon into `(nonce, rank, serve_port, hostname)`.
fn parse_beacon(s: &str) -> Option<(u64, u32, u16, String)> {
    let mut it = s.trim().splitn(5, '|');
    if it.next()? != BEACON_MAGIC {
        return None;
    }
    let nonce = u64::from_str_radix(it.next()?, 16).ok()?;
    let rank = it.next()?.parse().ok()?;
    let port = it.next()?.parse().ok()?;
    let host = it.next()?.to_string();
    Some((nonce, rank, port, host))
}

// ---- discovery driver ------------------------------------------------------

/// Discover ConnectX peers over `window`. `rank`/`serve_port` are advertised in
/// this node's beacon so peers can see them.
pub fn discover(rank: u32, serve_port: u16, window: Duration) -> Discovery {
    let dbg = std::env::var("COLI_DISC_DEBUG").is_ok();
    macro_rules! trace { ($($a:tt)*) => { if dbg { eprintln!("[disc] {}", format!($($a)*)); } } }
    let host = hostname();
    trace!("host={host}");
    let links = connectx_links();
    trace!("links={} (active {})", links.len(), links.iter().filter(|l| l.active).count());
    let local_ouis: Vec<String> = links.iter().map(ConnectXLink::oui).filter(|o| !o.is_empty()).collect();
    let local_ips: Vec<Ipv4Addr> = links.iter().filter_map(|l| l.ipv4).collect();

    let peers: Arc<Mutex<HashMap<Ipv4Addr, Peer>>> = Arc::new(Mutex::new(HashMap::new()));
    let nonce = make_nonce();
    let stop = Arc::new(AtomicBool::new(false));

    // Listener: collect beacons from other nodes for the window.
    let listener = UdpSocket::bind(("0.0.0.0", DISC_PORT)).ok();
    if let Some(sock) = &listener {
        let _ = sock.set_broadcast(true);
        let _ = sock.set_read_timeout(Some(Duration::from_millis(300)));
    }
    let ljoin = listener.as_ref().and_then(|s| s.try_clone().ok()).map(|sock| {
        let peers = peers.clone();
        let stop = stop.clone();
        let local_ips = local_ips.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 512];
            while !stop.load(Ordering::Relaxed) {
                match sock.recv_from(&mut buf) {
                    Ok((n, src)) => {
                        let msg = String::from_utf8_lossy(&buf[..n]);
                        if let Some((bn, brank, bport, bhost)) = parse_beacon(&msg) {
                            if bn == nonce {
                                continue; // our own broadcast
                            }
                            if let std::net::IpAddr::V4(ip) = src.ip() {
                                if local_ips.contains(&ip) {
                                    continue; // us via another local address
                                }
                                let mut m = peers.lock().unwrap();
                                let e = m.entry(ip).or_insert_with(|| Peer {
                                    ip,
                                    mac: None,
                                    hostname: None,
                                    rank: None,
                                    serve_port: None,
                                    via: String::new(),
                                    kind: PeerKind::FabricNeighbor,
                                    nic_match: false,
                                });
                                e.kind = PeerKind::Coli;
                                e.hostname = Some(bhost);
                                e.rank = Some(brank);
                                e.serve_port = Some(bport);
                            }
                        }
                    }
                    Err(_) => continue, // timeout tick
                }
            }
        })
    });

    // Broadcaster + sweeper, repeated across the window.
    let bsock = UdpSocket::bind("0.0.0.0:0").ok();
    if let Some(s) = &bsock {
        let _ = s.set_broadcast(true);
    }
    let msg = beacon_msg(nonce, rank, serve_port, &host);
    trace!("listener bound={} — entering {:?} window", listener.is_some(), window);
    let deadline = Instant::now() + window;
    let mut swept = false;
    while Instant::now() < deadline {
        if let Some(s) = &bsock {
            for l in links.iter().filter(|l| l.active) {
                if let Some(ip) = l.ipv4 {
                    let b = broadcast_addr(ip, l.prefix);
                    let _ = s.send_to(msg.as_bytes(), (b, DISC_PORT));
                }
            }
        }
        if !swept {
            sweep_subnets(&links);
            swept = true;
        }
        std::thread::sleep(Duration::from_millis(700));
    }
    trace!("window done — stopping listener");
    stop.store(true, Ordering::Relaxed);
    if let Some(j) = ljoin {
        let _ = j.join();
    }
    trace!("listener joined — reading ARP");

    // Fold in the ARP/neighbor table, classifying by NIC-OUI match. Only keep
    // neighbors that sit on one of our active link subnets.
    let subnets: Vec<(u32, u32, String)> = links
        .iter()
        .filter(|l| l.active)
        .filter_map(|l| {
            l.ipv4.map(|ip| {
                let mask = if l.prefix == 0 { 0 } else { u32::MAX << (32 - l.prefix as u32) };
                (u32::from(ip) & mask, mask, l.netdev.clone())
            })
        })
        .collect();
    {
        let mut m = peers.lock().unwrap();
        for (ip, mac, dev) in read_arp() {
            if local_ips.contains(&ip) {
                continue;
            }
            let ipn = u32::from(ip);
            let on_link = subnets.iter().find(|(net, mask, _)| ipn & mask == *net);
            let via = on_link.map(|(_, _, d)| d.clone()).unwrap_or(dev);
            if on_link.is_none() {
                continue; // not on a ConnectX subnet
            }
            let nic_match = local_ouis.contains(&oui_of(&mac));
            let e = m.entry(ip).or_insert_with(|| Peer {
                ip,
                mac: None,
                hostname: None,
                rank: None,
                serve_port: None,
                via: via.clone(),
                kind: PeerKind::FabricNeighbor,
                nic_match: false,
            });
            e.mac.get_or_insert(mac);
            if e.via.is_empty() {
                e.via = via;
            }
            e.nic_match = nic_match;
        }
    }

    trace!("merged ARP — finalizing");
    let map = match Arc::try_unwrap(peers) {
        Ok(m) => m.into_inner().unwrap(),
        Err(arc) => {
            // A stray reference is still alive — copy out rather than panic.
            arc.lock().unwrap().clone()
        }
    };
    let mut peers: Vec<Peer> = map.into_values().collect();
    peers.sort_by_key(|p| u32::from(p.ip));
    trace!("done — {} peers", peers.len());
    Discovery { hostname: host, links, peers }
}

/// Spawn a detached background beacon that broadcasts this node's presence on
/// every active ConnectX link every ~2 s, for the lifetime of the process. A
/// serving node calls this so peers that scan *later* still discover it (the
/// one-shot [`discover`] window only catches peers beaconing at the same time).
/// Returns the nonce it advertises (so the caller could correlate) — the thread
/// itself is left running.
pub fn spawn_beacon(rank: u32, serve_port: u16) -> u64 {
    let nonce = make_nonce();
    let host = hostname();
    let links = connectx_links();
    let msg = beacon_msg(nonce, rank, serve_port, &host);
    std::thread::spawn(move || {
        let sock = match UdpSocket::bind("0.0.0.0:0") {
            Ok(s) => s,
            Err(_) => return,
        };
        let _ = sock.set_broadcast(true);
        loop {
            for l in links.iter().filter(|l| l.active) {
                if let Some(ip) = l.ipv4 {
                    let b = broadcast_addr(ip, l.prefix);
                    let _ = sock.send_to(msg.as_bytes(), (b, DISC_PORT));
                }
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    });
    nonce
}

/// Render a human-readable report of the discovery result.
pub fn print_report(d: &Discovery, out: &mut impl std::io::Write) -> std::io::Result<()> {
    writeln!(out, "[cluster] node {} — ConnectX/RoCE fabric scan", d.hostname)?;
    if d.links.is_empty() {
        writeln!(out, "[cluster]   no RoCE devices found (/sys/class/infiniband empty) — not on a fabric")?;
        return Ok(());
    }
    writeln!(out, "[cluster]   local links:")?;
    for l in &d.links {
        let ip = l.ipv4.map(|i| format!("{i}/{}", l.prefix)).unwrap_or_else(|| "(no ip)".into());
        writeln!(
            out,
            "[cluster]     {:<12} {:<14} {:<18} {}  [{}]",
            l.rdma_dev,
            l.netdev,
            ip,
            l.mac,
            if l.active { "UP" } else { "down" }
        )?;
    }

    let sparks: Vec<&Peer> = d.peers.iter().filter(|p| p.kind == PeerKind::Coli || p.nic_match).collect();
    let coli = d.coli_peers().count();
    writeln!(
        out,
        "[cluster]   peers: {} Spark(s) on the fabric, {} running colibrì",
        sparks.len(),
        coli
    )?;
    for p in &d.peers {
        let is_spark = p.kind == PeerKind::Coli || p.nic_match;
        let tag = match (p.kind, is_spark) {
            (PeerKind::Coli, _) => "colibrì",
            (_, true) => "Spark NIC",
            (_, false) => "host",
        };
        let extra = match p.kind {
            PeerKind::Coli => format!(
                " — {} rank {} serving :{}",
                p.hostname.as_deref().unwrap_or("?"),
                p.rank.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
                p.serve_port.map(|s| s.to_string()).unwrap_or_else(|| "?".into())
            ),
            _ => String::new(),
        };
        writeln!(
            out,
            "[cluster]     {:<15} {:<18} via {:<14} {:<10}{}",
            p.ip,
            p.mac.as_deref().unwrap_or("(unknown mac)"),
            p.via,
            tag,
            extra
        )?;
    }
    if sparks.is_empty() {
        writeln!(out, "[cluster]   (no other Sparks seen — single node, or peers are down/not yet started)")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_addr_24() {
        assert_eq!(
            broadcast_addr(Ipv4Addr::new(192, 168, 100, 11), 24),
            Ipv4Addr::new(192, 168, 100, 255)
        );
        assert_eq!(
            broadcast_addr(Ipv4Addr::new(10, 0, 5, 3), 16),
            Ipv4Addr::new(10, 0, 255, 255)
        );
        assert_eq!(
            broadcast_addr(Ipv4Addr::new(172, 16, 4, 9), 30),
            Ipv4Addr::new(172, 16, 4, 11)
        );
    }

    #[test]
    fn beacon_roundtrip() {
        let msg = beacon_msg(0xdeadbeefcafef00d, 1, 8080, "gx10-5a4f");
        let (n, r, p, h) = parse_beacon(&msg).unwrap();
        assert_eq!(n, 0xdeadbeefcafef00d);
        assert_eq!(r, 1);
        assert_eq!(p, 8080);
        assert_eq!(h, "gx10-5a4f");
    }

    #[test]
    fn beacon_rejects_foreign() {
        assert!(parse_beacon("HELLO|x|y").is_none());
        assert!(parse_beacon("").is_none());
        assert!(parse_beacon("COLISPRK1|zz|1|8080|h").is_none()); // bad nonce
    }

    #[test]
    fn oui_extraction() {
        assert_eq!(oui_of("30:C5:99:40:42:B3"), "30:c5:99");
        assert_eq!(oui_of("30:c5:99:3f:5a:50"), "30:c5:99");
        // same OUI -> same hardware family (Spark NIC)
        assert_eq!(oui_of("30:C5:99:40:42:B3"), oui_of("30:c5:99:3f:5a:50"));
    }

    #[test]
    fn cidr_parse() {
        assert_eq!(parse_cidr("192.168.100.11/24"), Some((Ipv4Addr::new(192, 168, 100, 11), 24)));
        assert_eq!(parse_cidr("bogus"), None);
    }
}
