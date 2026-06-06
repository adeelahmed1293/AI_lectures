"""
Lab Assignment #02 - Part C: Deadlock Prevention via Resource Ordering
Course: Parallel and Distributed Computing Lab

Prevention Strategy: RESOURCE ORDERING
---------------------------------------
Assign a global numeric order to all resources:
    R1 = 1,  R2 = 2,  R3 = 3

Rule: A client may only request resource Rj if it holds NO resource Ri
      where i >= j.  In other words, requests must always be made in
      strictly increasing resource-order.

Why this eliminates deadlock:
    The Circular Wait condition (4th Coffman condition) is broken.
    If every client requests resources in the same ascending order,
    no cycle can form in the resource-allocation graph.
    Client 1 cannot hold R2 and wait for R1 — that would require
    requesting R1 (order 1) while holding R2 (order 2), violating the rule.

All four Coffman conditions:
    1. Mutual Exclusion  — still possible (R2 has only 1 unit)
    2. Hold and Wait     — still possible (clients hold while requesting)
    3. No Preemption     — server still never forcibly reclaims
    4. Circular Wait     — ELIMINATED by ordering policy ✓

Run: mpiexec -n 5 python part_c_deadlock_prevention.py
"""

from mpi4py import MPI
import time
import random

TAG_REQUEST = 10
TAG_REPLY   = 11
TAG_RELEASE = 12
TAG_DONE    = 99

# ---- Resource order (lower = higher priority / must be acquired first) ------
RESOURCE_ORDER = {"R1": 1, "R2": 2, "R3": 3}
RESOURCES      = {"R1": 2, "R2": 1, "R3": 2}


# =========================================================================== #
#  SERVER
# =========================================================================== #
def server(comm):
    size       = comm.Get_size()
    n_clients  = size - 1
    available  = dict(RESOURCES)
    allocated  = {r: {} for r in RESOURCES}
    done_count = 0

    print("\n[Server] Prevention Server started (Resource Ordering Policy).")
    print(f"[Server] Resource order: {RESOURCE_ORDER}")
    print(f"[Server] Initial pool:   {available}\n")
    print("[Server] Applying Resource Ordering Policy\n")

    while done_count < n_clients:
        status = MPI.Status()
        comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_REQUEST:
            msg      = comm.recv(source=src, tag=TAG_REQUEST)
            resource = msg["resource"]
            qty      = msg["qty"]
            highest_held = msg.get("highest_held", 0)   # order of highest resource client holds

            req_order = RESOURCE_ORDER[resource]

            print(f"[Server] Request from Client {src}: {qty}x {resource} "
                  f"(order={req_order}, client holds up to order={highest_held})")

            # ---- Resource Ordering Enforcement ---------------------------------
            if req_order <= highest_held:
                # Violates ordering rule — deny immediately to prevent deadlock
                comm.send({"status": "ORDER_VIOLATION",
                           "resource": resource, "qty": qty},
                          dest=src, tag=TAG_REPLY)
                print(f"[Server] ORDER VIOLATION: Client {src} requested {resource} "
                      f"(order {req_order}) while holding resource of order "
                      f"{highest_held}. Request DENIED to prevent deadlock.")
            elif available.get(resource, 0) >= qty:
                available[resource] -= qty
                allocated[resource][src] = allocated[resource].get(src, 0) + qty
                comm.send({"status": "GRANTED", "resource": resource, "qty": qty},
                          dest=src, tag=TAG_REPLY)
                print(f"[Server] GRANTED {qty}x {resource} to Client {src}. "
                      f"Available: {available[resource]}")
            else:
                comm.send({"status": "DENIED", "resource": resource, "qty": qty},
                          dest=src, tag=TAG_REPLY)
                print(f"[Server] DENIED {resource} to Client {src} "
                      f"(insufficient units: {available[resource]})")

        elif tag == TAG_RELEASE:
            msg      = comm.recv(source=src, tag=TAG_RELEASE)
            resource = msg["resource"]
            qty      = msg["qty"]
            available[resource] += qty
            if src in allocated[resource]:
                allocated[resource][src] -= qty
                if allocated[resource][src] <= 0:
                    del allocated[resource][src]
            print(f"[Server] Client {src} RELEASED {qty}x {resource}. "
                  f"Available: {available[resource]}")

        elif tag == TAG_DONE:
            comm.recv(source=src, tag=TAG_DONE)
            done_count += 1
            print(f"[Server] Client {src} finished ({done_count}/{n_clients}).")

    print("\n[Server] All clients done. Final pool:", available)
    print("[Server] No deadlock occurred — Resource Ordering Policy succeeded.")


# =========================================================================== #
#  CLIENT — Respects resource ordering
# =========================================================================== #
def client(comm):
    rank = comm.Get_rank()
    random.seed(rank * 13)

    # Each client picks 2 resources but ALWAYS requests them in ascending order
    chosen = random.sample(list(RESOURCE_ORDER.keys()), k=2)
    # Sort by resource order — this is the key prevention step
    chosen.sort(key=lambda r: RESOURCE_ORDER[r])

    print(f"[Client {rank}] Will request (in order): {chosen}")

    held = []    # resources currently held
    highest_order_held = 0

    for resource in chosen:
        qty = 1
        print(f"[Client {rank}] Requesting {resource} (order={RESOURCE_ORDER[resource]}) ...")
        comm.send({"resource": resource, "qty": qty,
                   "highest_held": highest_order_held},
                  dest=0, tag=TAG_REQUEST)
        reply = comm.recv(source=0, tag=TAG_REPLY)

        if reply["status"] == "GRANTED":
            held.append((resource, qty))
            highest_order_held = max(highest_order_held, RESOURCE_ORDER[resource])
            print(f"[Client {rank}] Holding {resource}. Simulating work ...")
            time.sleep(0.3 + rank * 0.05)
        elif reply["status"] == "ORDER_VIOLATION":
            print(f"[Client {rank}] Skipping {resource} — order violation prevented.")
        else:
            print(f"[Client {rank}] {resource} denied (unavailable).")

    # Release all held resources in reverse order (good practice)
    for resource, qty in reversed(held):
        comm.send({"resource": resource, "qty": qty}, dest=0, tag=TAG_RELEASE)
        print(f"[Client {rank}] Released {resource}.")

    comm.send(None, dest=0, tag=TAG_DONE)
    print(f"[Client {rank}] Finished successfully.")


# =========================================================================== #
#  Entry Point
# =========================================================================== #
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("ERROR: Need at least 2 MPI processes.")
        return

    if rank == 0:
        server(comm)
    else:
        client(comm)


if __name__ == "__main__":
    main()