#!/usr/bin/env python3
"""Parse a GGML_V100_DUMP_NODES stderr log into node records and answer structural queries.

Usage:
  python parse_nodes.py <log> [pass_index] [query]
queries: ops <ne0>          count ops for tensors with ne0 == value
         node <index>       print one node with its sources and their sources
         consumers <index>  print every node that reads node <index>
         list <ne0> <op>    list node indices with that ne0 and op
"""
import re
import sys
from collections import Counter

NODE_RE = re.compile(
    r"^\[cuda-node\]\s+(\d+)\s+(\S+)\s+(.*?)\s+(\S+)\s+"
    r"ne=\((\d+),(\d+),(\d+),(\d+)\)\s+flags=(\S+)$"
)
SRC_RE = re.compile(
    r"\[(\d+) (\S+) (\S+) (.*?) ne=\((\d+),(\d+),(\d+),(\d+)\)\]"
)


def parse(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"\[cuda-node\]\s+0 ", l)]
    starts.append(len(lines))
    passes = []
    for k in range(len(starts) - 1):
        blk = lines[starts[k]:starts[k + 1]]
        nd = {}
        for l in blk:
            m = NODE_RE.match(l.rstrip())
            if m:
                nd[int(m.group(1))] = {
                    "op": m.group(2), "name": m.group(3).strip(), "type": m.group(4),
                    "ne": tuple(int(m.group(i)) for i in (5, 6, 7, 8)),
                    "flags": m.group(9), "src": [],
                }
            elif l.startswith("[cuda-src ]"):
                i = int(re.match(r"\[cuda-src \]\s+(\d+) ", l).group(1))
                if i not in nd:
                    continue
                for sm in SRC_RE.finditer(l):
                    nd[i]["src"].append({
                        "slot": int(sm.group(1)), "op": sm.group(2), "type": sm.group(3),
                        "name": sm.group(4).strip(),
                        "ne": tuple(int(sm.group(j)) for j in (5, 6, 7, 8)),
                    })
        passes.append({"nodes": nd, "first": starts[k]})
    return passes


def desc(n):
    return "%s %-24s %-6s ne=%s" % (n["op"], n["name"][:24], n["type"], n["ne"])


def main():
    path = sys.argv[1]
    pi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    q = sys.argv[3] if len(sys.argv) > 3 else "summary"
    p = parse(path)[pi]
    nd = p["nodes"]
    print("pass %d: %d nodes" % (pi, len(nd)))

    if q == "summary":
        print(Counter(n["op"] for n in nd.values()).most_common(30))
    elif q == "ops":
        want = int(sys.argv[4])
        print(Counter(n["op"] for n in nd.values() if n["ne"][0] == want).most_common())
    elif q == "list":
        want, op = int(sys.argv[4]), sys.argv[5]
        for i in sorted(nd):
            if nd[i]["ne"][0] == want and nd[i]["op"] == op:
                print(i, desc(nd[i]))
    elif q == "node":
        i = int(sys.argv[4])
        n = nd[i]
        print(i, desc(n), "flags", n["flags"])
        for s in n["src"]:
            print("   src[%d] %s" % (s["slot"], desc(s)))
            if s["slot"] in nd:
                for s2 in nd[s["slot"]]["src"]:
                    print("        ^%s src[%d] %s" % (nd[s["slot"]]["op"], s2["slot"], desc(s2)))
    elif q == "consumers":
        i = int(sys.argv[4])
        for j in sorted(nd):
            for s in nd[j]["src"]:
                if s["slot"] == i:
                    print("node %d uses %d as src[%d]: %s" % (j, i, s["slot"], desc(nd[j])))
    else:
        print("unknown query")


if __name__ == "__main__":
    main()
