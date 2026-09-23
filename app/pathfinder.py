"""Whole-graph path construction with deterministic selection and a
complete rejection proof.

Search is iterative deepening on certificate count; at a fixed depth issuer
choices are visited in ascending fingerprint order. The first chain that
passes *every* gate (signature, validity, hierarchy, pathLen, name
constraints, policy, EKU, bitemporal revocation) at the smallest depth is the
unique required winner (fewest certs, lexicographically smallest
leaf→root fingerprint sequence). Revocation is checked on every complete
anchor path, never after fixing one shortest chain.

Exploration is lazy: from a certificate only name/AKI-compatible issuers are
considered, so 100k unrelated certificates add essentially zero work; cycles
are blocked with the on-path set, and all edge results are memoized.

Scalability for cross-signed ladders. Layers of DER-distinct certificates
that share subject name, key and superior fork the search combinatorially,
so no gate may cost O(depth) per concrete path:

* intrinsic node gates (validity/basic constraints/key usage) are per node;
* EKU and name-constraint verdicts (the leaf is fixed) are per certificate;
* revocation conclusions are per certificate (engine-side cache);
* pathLen/name/EKU/revocation state is carried incrementally down the
  leaf→root descent as a constant-size context, so each visited edge costs
  O(1) regardless of chain depth;
* RFC 5280 policy processing is the only inherently whole-path gate; it is
  evaluated on the policy-relevant projection (:class:`~app.chain.PolicyInput`)
  of a path, hence DER-equivalent cross-signs share one evaluation and the
  trace/failure is remapped to concrete fingerprints;
* static issuer resolution is computed once per node even though iterative
  deepening visits nodes repeatedly;
* a completed concrete path is finalized at most once across all deepening
  passes, and equivalent complete-path failures are aggregated in the
  rejection proof instead of materializing one record per DER permutation.
"""
from __future__ import annotations

import heapq

from .chain import (
    MAX_PATH_DEPTH,
    PolicyInput,
    eku_cert_violation,
    leaf_dns_names,
    leaf_uri_names,
    name_constraint_violation,
    process_policy_inputs,
)
from .graph import CertGraph

# Incremental gate context (leaf -> current node), fields:
#   pl_fail  : dict | None   first pathLen violation (nearest leaf first)
#   nc_fail  : dict | None   first name-constraint violation
#   eku_fail : dict | None   first EKU violation (leaf included at baseline)
#   ca_count : int           non-self-issued CA certs on the current stack
#   policy_proj : tuple      policy-relevant projection, leaf -> current
#
# Revocation is deliberately NOT carried here: its engine records an audit
# snapshot as a side effect, and the legacy semantics evaluate it only on
# complete anchor-terminated paths (nearest-leaf first), never on prefixes
# that fail to terminate or on longer branches popped before the winner.
from typing import NamedTuple


class _Ctx(NamedTuple):
    pl_fail: dict | None
    nc_fail: dict | None
    eku_fail: dict | None
    ca_count: int
    policy_proj: tuple


class PathFinder:
    def __init__(self, graph: CertGraph, anchors: set[str], revocation_eval,
                 signed_at: int, initial_policies: frozenset[str]):
        self.g = graph
        self.anchors = anchors
        self.rev = revocation_eval
        self.signed_at = signed_at
        self.initial_policies = initial_policies
        self.edge_failures: dict[tuple[str, str], dict] = {}
        self.node_failures: dict[str, dict] = {}
        # Aggregated complete-path failures: key (anchor, rule, at) -> group.
        self.path_groups: dict[tuple[str, str, str], dict] = {}
        # Smallest complete failing path overall (length, path) — the
        # deterministic terminal example the proof reports.
        self._terminal_example: tuple[tuple[str, ...], str, str] | None = None
        self.edges_seen: set[tuple[str, str]] = set()
        self._last_good: dict | None = None
        self._intrinsic_cache: dict[tuple[str, bool], dict | None] = {}
        # Per-find caches.
        self._completed: dict[tuple[str, ...], dict | None] = {}
        self._nc_cache: dict[str, dict | None] = {}
        self._eku_cache: dict[tuple[str, bool], dict | None] = {}
        self._policy_cache: dict[tuple, tuple] = {}
        self._policy_class: dict[str, tuple] = {}
        self._static_parents_cache: dict[str, list[str]] = {}
        self._dns_names: list[str] = []
        self._uri_names: list[str] = []

    # --------------------------------------------------------------- nodes
    def _node_intrinsic(self, fp: str, as_issuer: bool) -> dict | None:
        key = (fp, as_issuer)
        if key in self._intrinsic_cache:
            return self._intrinsic_cache[key]
        pc = self.g.get_cert(fp)
        failure = None
        if not (pc.not_before <= self.signed_at <= pc.not_after):
            failure = {"rule": "VALIDITY",
                       "detail": {"not_before": pc.not_before,
                                  "not_after": pc.not_after,
                                  "signed_at": self.signed_at}}
        elif as_issuer:
            if not pc.is_ca:
                failure = {"rule": "BASIC_CONSTRAINTS",
                           "detail": {"reason": "issuer certificate is not a CA"}}
            elif pc.key_usage and "keyCertSign" not in pc.key_usage:
                failure = {"rule": "KEY_USAGE",
                           "detail": {"reason": "issuer lacks keyCertSign"}}
        self._intrinsic_cache[key] = failure
        return failure

    # ------------------------------------------------- per-cert gate pieces
    def _rev_conclusion(self, fp: str) -> dict:
        return self.rev(self.g.get_cert(fp))

    def _nc_verdict(self, fp: str) -> dict | None:
        if fp not in self._nc_cache:
            self._nc_cache[fp] = name_constraint_violation(
                self.g.get_cert(fp), self._dns_names, self._uri_names)
        return self._nc_cache[fp]

    def _eku_verdict(self, fp: str, is_leaf: bool) -> dict | None:
        key = (fp, is_leaf)
        if key not in self._eku_cache:
            self._eku_cache[key] = eku_cert_violation(
                self.g.get_cert(fp), is_leaf=is_leaf)
        return self._eku_cache[key]

    def _policy_class_of(self, fp: str) -> tuple:
        cls = self._policy_class.get(fp)
        if cls is None:
            pc = self.g.get_cert(fp)
            cls = (pc.subject_der == pc.issuer_der,
                   pc.policies.oids,
                   pc.policy_mappings,
                   pc.require_explicit_policy,
                   pc.inhibit_policy_mapping,
                   pc.inhibit_any_policy)
            self._policy_class[fp] = cls
        return cls

    # ------------------------------------------------------------ contexts
    def _leaf_context(self, leaf_fp: str) -> _Ctx:
        pc = self.g.get_cert(leaf_fp)
        ca_count = 1 if pc.is_ca and pc.subject_der != pc.issuer_der else 0
        return _Ctx(pl_fail=None, nc_fail=None,
                    eku_fail=self._eku_verdict(leaf_fp, is_leaf=True),
                    ca_count=ca_count,
                    policy_proj=(self._policy_class_of(leaf_fp),))

    def _extend(self, ctx: _Ctx, ip: str) -> _Ctx:
        """Context after appending a non-anchor issuer ``ip`` (leaf→root).

        Only side-effect-free gates are carried; revocation is evaluated at
        completion (see :meth:`_finish_path`)."""
        pc = self.g.get_cert(ip)
        # PathLen: this issuer's bound applies to non-self CAs below it.
        pl_fail = ctx.pl_fail
        if pl_fail is None and pc.path_len is not None \
                and ctx.ca_count > pc.path_len:
            pl_fail = {"rule": "PATH_LEN", "at": ip,
                       "path_len": pc.path_len,
                       "counted_below": ctx.ca_count}
        ca_count = ctx.ca_count + (
            1 if pc.is_ca and pc.subject_der != pc.issuer_der else 0)
        # Name constraints / EKU: per-cert verdicts, first failure wins.
        nc_fail = ctx.nc_fail
        if nc_fail is None:
            nc_fail = self._nc_verdict(ip)
        eku_fail = ctx.eku_fail
        if eku_fail is None:
            eku_fail = self._eku_verdict(ip, is_leaf=False)
        return _Ctx(pl_fail=pl_fail, nc_fail=nc_fail, eku_fail=eku_fail,
                    ca_count=ca_count,
                    policy_proj=ctx.policy_proj + (self._policy_class_of(ip),))

    def _revocation_fail(self, fps: tuple[str, ...]) -> dict | None:
        """First non-GOOD certificate in leaf→root order (anchor excluded).

        Runs only on complete anchor paths, preserving the engine audit
        snapshot's exact contents."""
        for fp in fps[:-1]:
            res = self._rev_conclusion(fp)
            if res["conclusion"] != "GOOD":
                return {"rule": "REVOCATION", "at": fp,
                        "conclusion": res["conclusion"]}
        return None

    # -------------------------------------------------------------- policy
    def _policy_gate(self, path: tuple[str, ...],
                     proj_leaf_root: tuple) -> tuple[dict | None, list[dict] | None]:
        """Whole-path RFC 5280 policy processing on the projected chain.

        Equivalent DER cross-signs (identical projection) share one
        evaluation; the concrete trace is materialized only for an accepted
        winner. Fingerprints in the failure ``at`` are remapped to the
        concrete path. ``proj_leaf_root`` entries are fingerprint-free
        6-tuples (see :meth:`_policy_class_of`)."""
        # Memoize on the depth-ordered (root->leaf) projection.
        key = tuple(reversed(proj_leaf_root))
        cached = self._policy_cache.get(key)
        if cached is None:
            inputs = [
                PolicyInput(fingerprint=path[i],
                            self_issued=proj_leaf_root[i][0],
                            policy_oids=proj_leaf_root[i][1],
                            policy_mappings=proj_leaf_root[i][2],
                            require_explicit_policy=proj_leaf_root[i][3],
                            inhibit_policy_mapping=proj_leaf_root[i][4],
                            inhibit_any_policy=proj_leaf_root[i][5])
                for i in range(len(path))]
            ok, detail, trace = process_policy_inputs(inputs,
                                                      self.initial_policies)
            repr_fps = tuple(row["certificate"] for row in trace)
            at_depth = 0
            if detail is not None and detail.get("at") in repr_fps:
                at_depth = repr_fps.index(detail["at"]) + 1
            cached = (ok, detail, at_depth)
            self._policy_cache[key] = cached
        ok, detail, at_depth = cached
        if not ok:
            conc_detail = dict(detail)
            if at_depth:
                conc_detail["at"] = path[len(path) - at_depth]
            return conc_detail, None
        return None, self._concrete_policy_trace(path, proj_leaf_root)

    def _concrete_policy_trace(self, path: tuple[str, ...],
                               proj_leaf_root: tuple) -> list[dict]:
        """Rebuild the representative trace with concrete fingerprints for
        the selected path. ``process_policy_inputs`` takes leaf→root and
        reverses internally, so inputs stay in leaf→root order; output depth
        d (root→leaf) maps to path index n-d."""
        inputs = [
            PolicyInput(fingerprint=path[i],
                        self_issued=proj_leaf_root[i][0],
                        policy_oids=proj_leaf_root[i][1],
                        policy_mappings=proj_leaf_root[i][2],
                        require_explicit_policy=proj_leaf_root[i][3],
                        inhibit_policy_mapping=proj_leaf_root[i][4],
                        inhibit_any_policy=proj_leaf_root[i][5])
            for i in range(len(path))]
        _, _, trace = process_policy_inputs(inputs, self.initial_policies)
        n = len(path)
        out = []
        for row in trace:
            row2 = dict(row)
            row2["certificate"] = path[n - row2["depth"]]
            out.append(row2)
        return out

    # ----------------------------------------------------------------- API
    def find(self, leaf_fp: str) -> dict:
        self._completed = {}
        leaf_pc = self.g.get_cert(leaf_fp)
        if leaf_pc is None:
            return {"status": "REJECTED",
                    "reason": {"rule": "LEAF_NOT_IN_EVIDENCE_SET"},
                    "selected_path": None, "rejection_proof": None}
        self._dns_names = leaf_dns_names(leaf_pc)
        self._uri_names = leaf_uri_names(leaf_pc)
        lf = self._node_intrinsic(leaf_fp, as_issuer=False)
        if lf is not None:
            return self._reject(leaf_fp, lf)
        if leaf_fp in self.anchors:
            return {"status": "ACCEPTED", "reason": None,
                    "selected_path": [leaf_fp],
                    "policy_trace": [], "revocation": {},
                    "rejection_proof": None}

        base_ctx = self._leaf_context(leaf_fp)
        winner = self._enumerate(leaf_fp, base_ctx)
        if winner is not None:
            return {"status": "ACCEPTED", "reason": None,
                    "selected_path": winner["path"],
                    "policy_trace": winner["policy_trace"],
                    "revocation": winner["revocation"],
                    "rejection_proof": None,
                    "explored_edges": sorted(self.edges_seen),
                    "node_failures": [
                        {"certificate": fp, **f}
                        for (fp, as_issuer), f in self._intrinsic_cache.items()
                        if f is not None]}
        if self._terminal_example is not None:
            tpath, rule, at = self._terminal_example
            terminal = {"rule": rule,
                        "detail": {"at": at, "anchor": tpath[-1],
                                   "example_path": list(tpath)}}
        else:
            terminal = self.node_failures.get(leaf_fp) or {"rule": "NO_PATH_TO_ANCHOR"}
        return self._reject(leaf_fp, terminal)

    def _enumerate(self, leaf_fp: str, base_ctx: _Ctx) -> dict | None:
        """Enumerate anchor-terminated simple paths in the deterministic
        selection order: fewest certificates first, ties broken by the
        lexicographically smallest leaf→root fingerprint sequence.

        A best-first frontier ordered by (path length, path) yields exactly
        that order while visiting each concrete simple prefix once (no
        iterative-deepening re-traversal) and finalizing every complete
        candidate path, so the first acceptable complete path is the unique
        winner and the rejection proof covers all complete branches.
        Cycles are blocked by the on-path set; paths longer than
        MAX_PATH_DEPTH edges are not expanded (same bound as before)."""
        # Heap entries: (depth, stack_tuple, context). Ordering on
        # (len(stack), stack) is implemented directly in the heap tuple.
        heap: list[tuple[int, tuple[str, ...], _Ctx]] = []
        heapq.heappush(heap, (1, (leaf_fp,), base_ctx))
        while heap:
            depth, stack, ctx = heapq.heappop(heap)
            on_path = set(stack)
            fp = stack[-1]
            parents = self._static_parents(fp)
            # Push non-anchor extensions first; finalize anchors in order.
            anchor_hits: list[tuple[str, tuple[str, ...]]] = []
            extendable: list[tuple[str, tuple[str, ...], _Ctx]] = []
            for ip in parents:
                if ip in on_path:
                    self.edge_failures.setdefault((fp, ip), {"rule": "LOOP"})
                    continue
                nf = self._node_intrinsic(ip, as_issuer=True)
                if nf is not None:
                    self.edge_failures.setdefault((fp, ip), nf)
                    continue
                nxt = stack + (ip,)
                if ip in self.anchors:
                    anchor_hits.append((ip, nxt))
                elif depth < MAX_PATH_DEPTH:
                    extendable.append((ip, nxt, self._extend(ctx, ip)))
            # Anchors at this node: finalize in ascending fingerprint order;
            # the heap guarantees this node/path is itself in order.
            for ip, nxt in sorted(anchor_hits, key=lambda x: x[0]):
                if self._finish_path(nxt, ctx, ip) is None:
                    return self._last_good
            for ip, nxt, child_ctx in extendable:
                heapq.heappush(heap, (depth + 1, nxt, child_ctx))
        return None

    # ----------------------------------------------------------------- DFS
    def _static_parents(self, fp: str) -> list[str]:
        """Name/key- and signature-valid issuer fingerprints (sorted).

        Intrinsic to a node and independent of traversal, so it is resolved
        once even though iterative deepening visits the node many times."""
        cached = self._static_parents_cache.get(fp)
        if cached is not None:
            return cached
        out: list[str] = []
        for ip in self.g.candidate_issuers(fp):
            self.edges_seen.add((fp, ip))
            e = self.g.edge(fp, ip)
            if not e.name_key_ok:
                self.edge_failures.setdefault((fp, ip), {"rule": "ISSUER_NAME_KEY"})
            elif not e.sig_ok:
                self.edge_failures.setdefault((fp, ip),
                                              {"rule": e.sig_rule or "SIGNATURE"})
            else:
                out.append(ip)
        out = sorted(out)
        self._static_parents_cache[fp] = out
        return out

    def _finish_path(self, path: tuple[str, ...], ctx: _Ctx,
                     anchor_fp: str):
        """Apply every whole-path gate to one concrete anchor-terminated path
        and finalize it exactly once. Gate priority (revocation, pathLen,
        name constraints, EKU, policy) and the nearest-leaf-first scan order
        are part of the verdict semantics. Returns None iff acceptable."""
        cached = self._completed.get(path)
        if cached is not None or path in self._completed:
            return cached
        result: dict | None
        trace: list[dict] = []
        rev_fail = self._revocation_fail(path)
        if rev_fail is not None:
            result = rev_fail
        else:
            # Anchor participates in pathLen/name/EKU/policy (it constrains
            # everything below it) but is never revocation-checked.
            pc = self.g.get_cert(anchor_fp)
            pl_fail = ctx.pl_fail
            if pl_fail is None and pc.path_len is not None \
                    and ctx.ca_count > pc.path_len:
                pl_fail = {"rule": "PATH_LEN", "at": anchor_fp,
                           "path_len": pc.path_len,
                           "counted_below": ctx.ca_count}
            if pl_fail is not None:
                result = pl_fail
            else:
                nc_fail = ctx.nc_fail or self._nc_verdict(anchor_fp)
                if nc_fail is not None:
                    result = nc_fail
                else:
                    eku_fail = ctx.eku_fail or self._eku_verdict(
                        anchor_fp, is_leaf=False)
                    if eku_fail is not None:
                        result = eku_fail
                    else:
                        proj = ctx.policy_proj + (
                            self._policy_class_of(anchor_fp),)
                        result, trace = self._policy_gate(path, proj)
        self._completed[path] = result
        if result is None:
            self._last_good = {
                "path": list(path), "policy_trace": trace,
                "revocation": self._revocation_details(list(path))}
        else:
            self._record_path_failure(path, result)
        return result

    def _revocation_details(self, fps: list[str]) -> dict:
        details = {}
        for fp in fps[:-1]:
            res = self.rev(self.g.get_cert(fp))
            details[fp] = {"conclusion": res["conclusion"],
                           "selected_evidence": res["selected_evidence"]}
        return details

    def _record_path_failure(self, path: tuple[str, ...], detail: dict) -> None:
        rule = detail["rule"]
        at = detail.get("at", path[-1])
        anchor = path[-1]
        key = (anchor, rule, at)
        g = self.path_groups.get(key)
        example = list(path)
        if g is None:
            self.path_groups[key] = {"anchor": anchor, "rule": rule, "at": at,
                                     "path_count": 1, "example_path": example}
        else:
            g["path_count"] += 1
            # Representative: shortest failing path, then lexicographically
            # smallest leaf->root fingerprint sequence.
            if (len(example), example) < (len(g["example_path"]),
                                          g["example_path"]):
                g["example_path"] = example
        if (self._terminal_example is None
                or (len(path), path) < (len(self._terminal_example[0]),
                                        self._terminal_example[0])):
            self._terminal_example = (path, rule, at)
        edge = (path[-2], path[-1])
        self.edge_failures.setdefault(edge, {"rule": rule, "at": at,
                                             "path_level": True})

    # ------------------------------------------------------- rejection proof
    def _reject(self, leaf_fp: str, terminal: dict) -> dict:
        self._annotate_dead_frontiers()
        groups = sorted(self.path_groups.values(),
                        key=lambda x: (x["anchor"], x["rule"], x["at"],
                                       x["example_path"]))
        edges = [{"child": c, "parent": p,
                  "first_failure": self.edge_failures.get((c, p))}
                 for (c, p) in sorted(self.edges_seen)]
        proof = {
            "leaf": leaf_fp,
            "terminal_failure": terminal,
            "edges": edges,
            "node_failures": [{"certificate": fp, **f}
                              for fp, f in sorted(self.node_failures.items())],
            "path_level_failures": groups,
            "coverage": ("every name/key-compatible issuer edge cryptographically"
                         " considered from the leaf's reachable branch set"),
        }
        return {"status": "REJECTED", "reason": terminal,
                "selected_path": None, "rejection_proof": proof}

    def _annotate_dead_frontiers(self) -> None:
        """Edges intrinsically valid but leading to a branch that can never
        terminate at an anchor get NO_PATH_TO_ANCHOR, so the proof covers all
        reachable candidate branches rather than only the final attempt."""
        # Static adjacency among explored nodes.
        parents_of: dict[str, set[str]] = {}
        for (c, p) in self.edges_seen:
            parents_of.setdefault(c, set()).add(p)
        can = set(self.anchors)
        children_of: dict[str, set[str]] = {}
        for c, ps in parents_of.items():
            for p in ps:
                children_of.setdefault(p, set()).add(c)
        stack = list(can)
        seen = set(can)
        while stack:
            p = stack.pop()
            for c in children_of.get(p, ()):  # only explored edges
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        for (c, p) in self.edges_seen:
            if self.edge_failures.get((c, p)):
                continue
            e = self.g.edge(c, p)
            if e.name_key_ok and e.sig_ok and p not in seen:
                self.edge_failures[(c, p)] = {"rule": "NO_PATH_TO_ANCHOR"}
        # Merge node_failures from cache (issuer-side failures only).
        for (fp, as_issuer), fail in self._intrinsic_cache.items():
            if fail is not None:
                self.node_failures.setdefault(fp, fail)
