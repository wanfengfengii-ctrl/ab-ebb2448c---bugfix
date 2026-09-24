"""Whole-graph path construction with deterministic selection and a
complete rejection proof.

Search is iterative deepening on certificate count; at a fixed depth issuer
choices are visited in ascending fingerprint order. The first chain that
passes *every* gate (signature, validity, hierarchy, pathLen, name
constraints, EKU, bitemporal revocation, policy) at the smallest depth is the
unique required winner (fewest certs, lexicographically smallest
leaf→root fingerprint sequence). Revocation is resolved per certificate
(the revocation universe is fixed for an adjudication) and checked on every
complete anchor path.

Exploration is lazy: from a certificate only name/AKI-compatible issuers are
considered, so 100k unrelated certificates add essentially zero work, and
cycles are bounded by ``MAX_PATH_DEPTH`` plus the on-path fingerprint set.

Equivalent-branch merging
-------------------------
Cross-signed CAs commonly arrive as several DER-distinct certificates that
are identical for every validation decision: same validity window, CA/KU
bits, pathLen, name constraints, EKU, policy content and revocation
conclusion, and (by greatest-fixed-point partition refinement) the same set
of issuer *equivalence classes*. Such certificates are bisimilar: a path
through one variant has exactly the same gate outcomes as a path through
another. Enumerating all 2^L variants (L cross-signed layers) and re-running
path-level gates per concrete path is exponential in L while adding zero
information.

Path enumeration therefore runs over equivalence classes (one node per
layer in a cross-signed ladder). Gate evaluation happens once per complete
*class* path; concrete fingerprints are retained per class and expanded only
for (a) the deterministic winner (lexicographically smallest fingerprint
sequence) and (b) rejection coverage (an example path plus the number of
concrete paths each merged failure covers). Every cryptographically
considered edge is still listed individually in the rejection proof.
"""
from __future__ import annotations

from .chain import (
    MAX_PATH_DEPTH,
    OID_ANY_POLICY,
    _PolicyTreeNode,
    check_eku,
    check_name_constraints,
    check_path_len,
    policy_step,
    policy_wrap_up,
    process_policies,
)
from .graph import CertGraph

_ROOT_POLICY_LEVEL = (_PolicyTreeNode(OID_ANY_POLICY, (OID_ANY_POLICY,)),)


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
        self.path_outcomes: list[dict] = []
        self.edges_seen: set[tuple[str, str]] = set()
        self._accepted_seqs: list[tuple[int, ...]] = []
        self._intrinsic_cache: dict[tuple[str, bool], dict | None] = {}
        self._parents_cache: dict[str, list[str]] = {}
        # Memoized RFC 5280 policy transitions shared with chain.py.
        self._content_rep: dict[tuple, object] = {}
        self._step_cache: dict[tuple, tuple] = {}
        self._wrap_cache: dict[tuple, tuple] = {}

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

    def _revocation_conclusion(self, pc) -> str:
        # Cached inside the revocation engine; only the conclusion feeds a
        # gate decision, so it is the only revocation datum in a class label.
        return self.rev(pc)["conclusion"]

    def _revocation_gate(self, pcs: list) -> tuple[dict | None, dict]:
        details = {}
        # Trust anchor revocation status is not evaluated (RFC 5280 §6.1.3).
        for pos, pc in enumerate(pcs[:-1]):
            res = self.rev(pc)
            details[pc.fingerprint] = {"conclusion": res["conclusion"],
                                       "selected_evidence": res["selected_evidence"]}
            if res["conclusion"] != "GOOD":
                return {"rule": "REVOCATION", "_pos": pos,
                        "detail": {"at": pc.fingerprint,
                                   "conclusion": res["conclusion"]}}, details
        return None, details

    # ----------------------------------------------------------------- API
    def find(self, leaf_fp: str) -> dict:
        self._intrinsic_cache = {}
        if self.g.get_cert(leaf_fp) is None:
            return {"status": "REJECTED",
                    "reason": {"rule": "LEAF_NOT_IN_EVIDENCE_SET"},
                    "selected_path": None, "rejection_proof": None}
        lf = self._node_intrinsic(leaf_fp, as_issuer=False)
        if lf is not None:
            return self._reject(leaf_fp, lf)
        if leaf_fp in self.anchors:
            return {"status": "ACCEPTED", "reason": None,
                    "selected_path": [leaf_fp],
                    "policy_trace": [], "revocation": {},
                    "rejection_proof": None}

        classes = self._build_classes(leaf_fp)
        if classes is None:
            # No statically valid branch reaches anything at all.
            terminal = self.node_failures.get(leaf_fp) or {"rule": "NO_PATH_TO_ANCHOR"}
            return self._reject(leaf_fp, terminal)

        leaf_cls = classes.fp_class[leaf_fp]
        winner = None
        for depth in range(1, MAX_PATH_DEPTH + 1):
            self._accepted_seqs: list[tuple[int, ...]] = []
            self._dfs_classes(classes, leaf_cls, (leaf_cls,),
                              {leaf_cls: 1}, depth, leaf_fp)
            if self._accepted_seqs:
                # Deterministic winner at this (smallest) depth: fewest
                # certificates is fixed by the depth; break ties on the
                # lexicographically smallest concrete leaf→root fingerprint
                # sequence, resolved per class sequence by the same solver.
                best_path: tuple[str, ...] | None = None
                for seq in self._accepted_seqs:
                    _count, real = self._solve_realizations(
                        classes, seq, leaf_fp)
                    if real is not None and (
                            best_path is None or real < best_path):
                        best_path = real
                path = list(best_path)
                pcs = [self.g.get_cert(fp) for fp in path]
                _rev_fail, rev_details = self._revocation_gate(pcs)
                assert _rev_fail is None
                ok, _detail, trace = process_policies(
                    pcs, self.initial_policies)
                assert ok
                winner = {"path": path, "policy_trace": trace,
                          "revocation": rev_details}
                break
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
        if self.path_outcomes:
            # Complete anchor-terminated paths existed but all failed a
            # whole-path gate; report the failure on the deterministic
            # smallest (length, fingerprint) such path.
            rep = sorted(self.path_outcomes,
                         key=lambda o: (len(o["path"]), o["path"]))[0]
            terminal = {"rule": rep["rule"],
                        "detail": {"at": rep["at"], "anchor": rep["anchor"],
                                   "example_path": rep["path"]}}
        else:
            terminal = self.node_failures.get(leaf_fp) or {"rule": "NO_PATH_TO_ANCHOR"}
        return self._reject(leaf_fp, terminal)

    # ------------------------------------------------- graph + partitioning
    def _static_parents(self, fp: str) -> list[str]:
        """Name/key- and signature-valid issuer fingerprints (sorted).

        Path-independent, so adjacency and signature verification happen once
        per certificate.
        """
        cached = self._parents_cache.get(fp)
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
        self._parents_cache[fp] = out
        return out

    def _gate_label(self, fp: str, rev_conclusion: str | None) -> tuple:
        """All gate-relevant content of a certificate.

        Two certificates with equal labels (and bisimilar issuer classes)
        are interchangeable in every path decision: same identity
        (subject/key/issuer), same validity and hierarchy bits, same
        constraints, EKU and policy content, and the same revocation
        conclusion — i.e. cross-certificates differing only in DER.
        """
        pc = self.g.get_cert(fp)
        return (
            fp in self.anchors,
            pc.subject_der,
            pc.issuer_der,
            pc.spki_bitstring,
            pc.aki,
            pc.ski,
            pc.not_before, pc.not_after,
            pc.is_ca,
            tuple(sorted(pc.key_usage or ())),
            pc.path_len,
            tuple(sorted(pc.eku or ())),
            tuple(sorted(pc.san_dns)),
            tuple(sorted(pc.san_uri)),
            tuple(sorted(pc.nc_permitted_dns)),
            tuple(sorted(pc.nc_excluded_dns)),
            tuple(sorted(pc.nc_permitted_uri)),
            tuple(sorted(pc.nc_excluded_uri)),
            tuple(sorted(pc.policies.oids)),
            tuple(sorted(pc.policy_mappings)),
            pc.require_explicit_policy,
            pc.inhibit_policy_mapping,
            pc.inhibit_any_policy,
            rev_conclusion,
        )

    def _build_classes(self, leaf_fp: str):
        """Closure over statically valid edges + greatest-fixed-point
        partition into gate-bisimilar certificate classes.

        Only certificates that can occur on a complete anchor-terminated
        path are partitioned; revocation is evaluated for exactly those
        (matching the engine snapshot's historical scope). Dead branches
        remain recorded in ``edges_seen`` for the rejection proof.
        """
        # 1. Forward BFS from the leaf over static + intrinsically valid
        #    edges (anchors terminate), keeping shortest edge distance. No
        #    certificate on an adjudicated path is farther than
        #    MAX_PATH_DEPTH from the leaf.
        fdist: dict[str, int] = {leaf_fp: 0}
        frontier = {leaf_fp}
        for step in range(1, MAX_PATH_DEPTH + 1):
            nxt: set[str] = set()
            for fp in frontier:
                for ip in self._static_parents(fp):
                    if ip in self.anchors:
                        # Trust anchors terminate, are never traversed and
                        # are not issuer-intrinsic-gated.
                        fdist.setdefault(ip, step)
                        continue
                    nf = self._node_intrinsic(ip, as_issuer=True)
                    if nf is not None:
                        self.edge_failures.setdefault((fp, ip), nf)
                        continue
                    if ip not in fdist:
                        fdist[ip] = step
                        nxt.add(ip)
            if not nxt:
                break
            frontier = nxt

        # 2. Backward BFS: shortest edge distance to any anchor.
        parents = {fp: {p for p in self._static_parents(fp) if p in fdist}
                   for fp in fdist if fp not in self.anchors}
        children_of: dict[str, set[str]] = {}
        for ch, ps in parents.items():
            for p in ps:
                children_of.setdefault(p, set()).add(ch)
        bdist: dict[str, int] = {a: 0 for a in self.anchors if a in fdist}
        frontier = set(bdist)
        step = 0
        while frontier and step < MAX_PATH_DEPTH:
            step += 1
            nxt = set()
            for ip in frontier:
                for ch in children_of.get(ip, ()):
                    if ch not in bdist:
                        bdist[ch] = step
                        nxt.add(ch)
            frontier = nxt

        # A certificate can occur on a completed adjudicated path iff some
        # leaf->anchor walk through it fits MAX_PATH_DEPTH; with shortest
        # distances on each side that is exactly fdist + bdist <= bound.
        live = {fp for fp in fdist if fp in bdist
                and fdist[fp] + bdist[fp] <= MAX_PATH_DEPTH}
        if leaf_fp not in live:
            return None

        # 3. Revocation conclusions are a label component; each distinct
        #    live non-anchor certificate is evaluated once (engine caches).
        rev_of: dict[str, str] = {}
        for fp in live:
            if fp not in self.anchors:
                rev_of[fp] = self._revocation_conclusion(self.g.get_cert(fp))

        # 4. Partition refinement over the live subgraph: equal gate label
        #    and equal set of issuer classes (anchor edges terminate, so an
        #    anchor's own issuers do not participate).
        labels = {fp: self._gate_label(fp, rev_of.get(fp)) for fp in live}
        parents = {fp: {p for p in parents.get(fp, ()) if p in live}
                   for fp in live}
        class_of: dict[str, int] = {}

        def repartition() -> None:
            groups: dict[tuple, list[str]] = {}
            for fp in live:
                if fp in self.anchors:
                    parent_classes: tuple = ()
                else:
                    parent_classes = tuple(sorted(
                        class_of[p] for p in parents[fp] if p in class_of))
                groups.setdefault((labels[fp], parent_classes), []).append(fp)
            new_class: dict[str, int] = {}
            for cid, ms in enumerate(sorted(groups.values(),
                                            key=lambda ms: min(ms))):
                for m in ms:
                    new_class[m] = cid
            class_of.clear()
            class_of.update(new_class)

        repartition()
        prev = None
        while prev != class_of:
            prev = dict(class_of)
            repartition()

        members: dict[int, list[str]] = {}
        for fp, cid in class_of.items():
            members.setdefault(cid, []).append(fp)
        for ms in members.values():
            ms.sort()
        class_parents: dict[int, list[int]] = {}
        anchor_classes: set[int] = set()
        for cid, ms in members.items():
            pcs = {class_of[p] for m in ms for p in parents.get(m, ())}
            class_parents[cid] = sorted(pcs, key=lambda c: members[c][0])
            if any(m in self.anchors for m in ms):
                anchor_classes.add(cid)
        return _ClassView(members, class_of, self._parents_cache,
                          class_parents, anchor_classes)

    # ------------------------------------------------------- class DFS
    def _dfs_classes(self, cv, cls: int, stack: tuple[int, ...],
                     visits: dict[int, int], remaining: int,
                     leaf_fp: str) -> None:
        """Enumerate every class walk of exactly ``remaining + 1`` nodes
        from the leaf class, and gate every one that terminates at an
        anchor. A chain may terminate only when exactly one node is left, so
        a short chain is gated once at its own depth instead of being
        repeated at every larger depth.

        Loop guard: the concrete search forbids repeating a fingerprint.
        Certificates in one class share subject and key but are DER-distinct,
        so a class may be revisited only while distinct members remain
        (visits < class size); beyond that the walk is a cycle and is
        dropped exactly as the fingerprint guard would. All terminating
        walks at a depth are visited, so rejection coverage stays complete
        even when an earlier walk would accept."""
        for pcls in cv.class_parents[cls]:
            is_anchor = pcls in cv.anchor_classes
            if remaining == 1:
                if not is_anchor:
                    continue
                seq = stack + (pcls,)
                if self._complete_class_path(cv, seq, leaf_fp):
                    self._accepted_seqs.append(seq)
                continue
            if is_anchor:
                continue
            nvisits = visits.get(pcls, 0)
            if nvisits >= len(cv.members[pcls]):
                continue
            visits[pcls] = nvisits + 1
            self._dfs_classes(cv, pcls, stack + (pcls,), visits,
                              remaining - 1, leaf_fp)
            visits[pcls] = nvisits

    def _representative_pcs(self, cv, seq: tuple[int, ...]) -> list:
        return [self.g.get_cert(cv.members[c][0]) for c in seq]

    def _solve_realizations(self, cv, seq: tuple[int, ...], leaf_fp: str,
                            pins: dict[int, str] | None = None):
        """Count concrete realizations of a class path and return the
        lexicographically smallest one.

        ``pins`` forces concrete members at given positions (used to split a
        merged failure by its concrete failure location and anchor). A
        realization requires a valid static edge between successive
        fingerprints, ends on a trust anchor, and never repeats a
        fingerprint (the concrete loop guard).

        A fingerprint belongs to exactly one class, so a repeat is possible
        only at positions whose class recurs in ``seq``. The DP therefore
        tracks used fingerprints solely for recurring classes; for the
        common non-recurring case the memo state is just
        ``(position, predecessor)`` and the cost is linear in path length no
        matter how many variants each class has.
        """
        pins = pins or {}
        layers = [sorted(cv.members[c]) for c in seq]
        recurring_positions: dict[int, list[int]] = {}
        for k, cid in enumerate(seq):
            recurring_positions.setdefault(cid, []).append(k)
        recurring_classes = {c for c, ps in recurring_positions.items()
                             if len(ps) > 1}

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def solve(k: int, prev_fp: str,
                  used: frozenset[str]) -> tuple[int, tuple[str, ...] | None]:
            if k == len(layers):
                return 1, ()
            options = layers[k]
            if k in pins:
                forced = pins[k]
                options = [forced] if forced in options else []
            best: tuple[str, ...] | None = None
            total = 0
            cls_here = seq[k]
            for m in options:
                if m in used or m not in cv.parents_of[prev_fp]:
                    continue
                if k == len(layers) - 1 and m not in self.anchors:
                    continue
                next_used = used | ({m} if cls_here in recurring_classes
                                    else frozenset())
                cnt, suffix = solve(k + 1, m, next_used)
                if cnt == 0:
                    continue
                total += cnt
                cand = (m,) + suffix
                if best is None or cand < best:
                    best = cand
            return total, best

        assert leaf_fp in layers[0]
        if 0 in pins and pins[0] != leaf_fp:
            return 0, None
        init_used = frozenset((leaf_fp,)) if seq[0] in recurring_classes \
            else frozenset()
        count, suffix = solve(1, leaf_fp, init_used)
        if count == 0:
            return 0, None
        return count, (leaf_fp,) + suffix

    def _complete_class_path(self, cv, seq: tuple[int, ...],
                             leaf_fp: str) -> bool:
        """Run every whole-path gate once for a class path, using the
        smallest-fingerprint member of each class. Members share gate
        labels, so every concrete realization has the same gate result.
        Rejected realizations are grouped by (anchor, failure location) with
        counts; the accepted realization is the lex-smallest concrete path.
        """
        rep_pcs = self._representative_pcs(cv, seq)
        rep_fps = [pc.fingerprint for pc in rep_pcs]

        def fail(rule: str, detail: dict, at_pos: int) -> bool:
            # Split the merged failure by concrete failure location and
            # anchor; count realizations per group via the memoized solver.
            groups: dict[tuple[str, str], tuple[int, tuple]] = {}
            at_members = cv.members[seq[at_pos]]
            anchor_members = cv.members[seq[-1]]
            for at_fp in sorted(at_members):
                for anchor_fp in sorted(anchor_members):
                    if at_pos == len(seq) - 1 and at_fp != anchor_fp:
                        continue
                    pins = {at_pos: at_fp, len(seq) - 1: anchor_fp}
                    count, example = self._solve_realizations(
                        cv, seq, leaf_fp, pins)
                    if count == 0:
                        continue
                    key = (anchor_fp, at_fp)
                    cur = groups.get(key)
                    if cur is None or example < cur[1]:
                        groups[key] = (count, example)
            if not groups:
                return False  # no loop-free concrete realization exists
            for (_anchor_fp, at_fp), (count, example) in groups.items():
                concrete = {k: v for k, v in detail.items() if k != "_pos"}
                concrete["at"] = at_fp
                self.path_outcomes.append({
                    "path": list(example), "rule": rule,
                    "at": at_fp, "anchor": example[-1],
                    "path_count": count})
                self.edge_failures.setdefault(
                    (example[-2], example[-1]),
                    {"rule": rule, "at": at_fp, "path_level": True})
            return False

        rev_fail, _details = self._revocation_gate(rep_pcs)
        if rev_fail is not None:
            return fail(rev_fail["rule"], rev_fail["detail"],
                        rev_fail.get("_pos", len(seq) - 1))
        for check in (check_path_len, check_name_constraints, check_eku):
            ok, detail = check(rep_pcs)
            if not ok:
                rule = detail.pop("rule")
                at_fp = detail.get("at", rep_fps[-1])
                # Gates fail on the first violating position leaf→root; when
                # a class recurs, identical members mean the first occurrence
                # is exactly the one the gate reports.
                at_pos = rep_fps.index(at_fp) if at_fp in rep_fps \
                    else len(seq) - 1
                return fail(rule, detail, at_pos)
        ppos = self._policy_failure(rep_pcs)
        if ppos is not None:
            at_pos, pdetail = ppos
            return fail("POLICY", pdetail, at_pos)

        # Gate-invariant acceptance: a concrete loop-free realization must
        # exist (the class walk itself never revisits a class, so this is the
        # usual case). The winner path is selected afterwards in find().
        count, _path = self._solve_realizations(cv, seq, leaf_fp)
        return count > 0

    # -------------------------------------------------------- policy walk
    def _policy_step(self, pc, i: int, counters: tuple[int, int, int],
                     level: tuple[_PolicyTreeNode, ...]):
        """Memoized :func:`chain.policy_step`, shared across equivalent
        cross-certificates (same policy content regardless of DER)."""
        content = self._policy_content(pc)
        self._content_rep.setdefault(content, pc)
        rep = self._content_rep[content]
        key = (content, i, counters, level)
        cached = self._step_cache.get(key)
        if cached is None:
            cached = policy_step(rep, i, counters, level)
            self._step_cache[key] = cached
        return cached

    @staticmethod
    def _policy_content(pc) -> tuple:
        return (
            pc.subject_der == pc.issuer_der,
            tuple(sorted(pc.policies.oids)),
            tuple(sorted(pc.policy_mappings)),
            pc.require_explicit_policy,
            pc.inhibit_policy_mapping,
            pc.inhibit_any_policy,
        )

    def _policy_failure(self, parsed_leaf_to_root: list):
        """Return ``(position, failure_detail)`` on failure (position is the
        leaf→root index of the failing certificate), else ``None``."""
        n = len(parsed_leaf_to_root)
        level = _ROOT_POLICY_LEVEL
        counters = (n, n, n + 1)
        for i, pc in enumerate(reversed(parsed_leaf_to_root), start=1):
            failure, result = self._policy_step(pc, i, counters, level)
            if failure is not None:
                return n - i, failure
            level, counters = result
        ok, detail = policy_wrap_up(level, counters[0], self.initial_policies)
        if not ok:
            return 0, detail
        return None

    # ------------------------------------------------------- rejection proof
    def _reject(self, leaf_fp: str, terminal: dict) -> dict:
        self._annotate_dead_frontiers()
        groups: dict[tuple, dict] = {}
        for o in self.path_outcomes:
            key = (o["anchor"], o["rule"], o["at"])
            g = groups.setdefault(key, {"anchor": o["anchor"], "rule": o["rule"],
                                        "at": o["at"], "path_count": 0,
                                        "example_path": o["path"]})
            g["path_count"] += o.get("path_count", 1)
            if o["path"] < g["example_path"]:
                g["example_path"] = o["path"]
        edges = [{"child": c, "parent": p,
                  "first_failure": self.edge_failures.get((c, p))}
                 for (c, p) in sorted(self.edges_seen)]
        proof = {
            "leaf": leaf_fp,
            "terminal_failure": terminal,
            "edges": edges,
            "node_failures": [{"certificate": fp, **f}
                              for fp, f in sorted(self.node_failures.items())],
            "path_level_failures": sorted(
                groups.values(),
                key=lambda x: (x["anchor"], x["rule"], x["at"], x["example_path"])),
            "coverage": ("every name/key-compatible issuer edge cryptographically"
                         " considered from the leaf's reachable branch set;"
                         " path_level_failures.path_count counts concrete paths"
                         " covered after merging DER-distinct cross-certificates"
                         " with identical gate content"),
        }
        return {"status": "REJECTED", "reason": terminal,
                "selected_path": None, "rejection_proof": proof}

    def _annotate_dead_frontiers(self) -> None:
        """Edges intrinsically valid but leading to a branch that can never
        terminate at an anchor get NO_PATH_TO_ANCHOR, so the proof covers all
        reachable candidate branches rather than only the final attempt."""
        parents_of: dict[str, set[str]] = {}
        for (c, p) in self.edges_seen:
            parents_of.setdefault(c, set()).add(p)
        children_of: dict[str, set[str]] = {}
        for c, ps in parents_of.items():
            for p in ps:
                children_of.setdefault(p, set()).add(c)
        stack = list(self.anchors)
        seen = set(self.anchors)
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


class _ClassView:
    """Gate-bisimilar partition of the reachable certificates."""

    def __init__(self, members, fp_class, parents_of, class_parents,
                 anchor_classes):
        self.members = members                 # cid -> sorted fingerprints
        self.fp_class = fp_class               # fp -> cid
        self.parents_of = parents_of           # fp -> set(parent fp)
        self.class_parents = class_parents     # cid -> sorted parent cids
        self.anchor_classes = anchor_classes
