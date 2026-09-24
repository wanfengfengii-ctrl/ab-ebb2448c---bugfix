"""RFC 5280 chain-level validation over a completed candidate path.

The implemented profile is documented in README ("Certificate profile").
Checks here operate on parsed-and-cryptographically-verified certificates
only. Paths are supplied leaf -> root; policy processing runs root -> leaf.
"""
from __future__ import annotations

from urllib.parse import urlparse

from .certmodel import ParsedCert
from .errors import OID_ANY_POLICY, OID_EKU_CODE_SIGNING

# Deterministic rule identifiers used in intermediate conclusions and in
# rejection proofs.
RULE_ORDER = [
    "PROFILE",
    "VALIDITY",
    "ISSUER_NAME_KEY",
    "SIGNATURE",
    "BASIC_CONSTRAINTS",
    "KEY_USAGE",
    "PATH_LEN",
    "NAME_CONSTRAINTS",
    "POLICY",
    "EKU",
    "REVOCATION",
    "ANCHOR",
    "LOOP",
    "PATH_DEPTH",
]

MAX_PATH_DEPTH = 20


def dns_in_subtree(host: str, constraint: str) -> bool:
    host = host.lower().rstrip(".")
    c = constraint.lower().lstrip(".").rstrip(".")
    if c == "":
        return True  # "." subtree matches every DNS name
    return host == c or host.endswith("." + c)


def uri_host(uri: str) -> str | None:
    try:
        u = urlparse(uri.lower())
        return u.hostname
    except ValueError:
        return None


def uri_in_subtree(uri: str, constraint: str) -> bool:
    host = uri_host(uri)
    if host is None:
        return False
    c = constraint.lower()
    if "://" in c or c.startswith("."):
        chost = uri_host(c if "://" in c else "http://" + c.lstrip("."))
        if chost is None:
            return False
        return dns_in_subtree(host, chost)
    # A bare constraint value is matched against the host component.
    return dns_in_subtree(host, c)


def _leaf_dns_names(leaf: ParsedCert) -> list[str]:
    names = list(leaf.san_dns)
    if not names:
        # Legacy: treat subject CNs as DNS names when constrained (RFC 5280
        # §4.2.1.10 guidance).
        from cryptography.x509.oid import NameOID

        for attr in leaf.cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME):
            if isinstance(attr.value, str):
                try:
                    names.append(attr.value.encode("ascii").decode().lower())
                except UnicodeEncodeError:
                    # non-ASCII CN cannot be matched by a DNS constraint
                    names.append("\x00-non-ascii-")
    return names


def _leaf_uri_names(leaf: ParsedCert) -> list[str]:
    return list(leaf.san_uri)


def check_name_constraints(path_leaf_to_root: list[ParsedCert]) -> tuple[bool, dict | None]:
    """Every CA's constraints apply to all certificates below it."""
    leaf = path_leaf_to_root[0]
    dns_names = _leaf_dns_names(leaf)
    uri_names = _leaf_uri_names(leaf)
    # path[0] leaf, path[1:] issuers; only issuers carry constraints that
    # constrain the leaf. (A self-issued leaf constraints are irrelevant.)
    for ca in path_leaf_to_root[1:]:
        for name in dns_names:
            if ca.nc_excluded_dns and any(dns_in_subtree(name, c) for c in ca.nc_excluded_dns):
                return False, {"rule": "NAME_CONSTRAINTS", "at": ca.fingerprint,
                               "kind": "dns_excluded", "name": name}
            if ca.nc_permitted_dns and not any(
                    dns_in_subtree(name, c) for c in ca.nc_permitted_dns):
                return False, {"rule": "NAME_CONSTRAINTS", "at": ca.fingerprint,
                               "kind": "dns_not_permitted", "name": name}
        for name in uri_names:
            if ca.nc_excluded_uri and any(uri_in_subtree(name, c) for c in ca.nc_excluded_uri):
                return False, {"rule": "NAME_CONSTRAINTS", "at": ca.fingerprint,
                               "kind": "uri_excluded", "name": name}
            if ca.nc_permitted_uri and not any(
                    uri_in_subtree(name, c) for c in ca.nc_permitted_uri):
                return False, {"rule": "NAME_CONSTRAINTS", "at": ca.fingerprint,
                               "kind": "uri_not_permitted", "name": name}
        if not dns_names and not uri_names and (
                ca.nc_permitted_dns or ca.nc_permitted_uri):
            return False, {"rule": "NAME_CONSTRAINTS", "at": ca.fingerprint,
                           "kind": "no_name_matching_permitted_subtree"}
    return True, None


def check_path_len(path_leaf_to_root: list[ParsedCert]) -> tuple[bool, dict | None]:
    """pathLenConstraint in CA cert C limits non-self-issued CA certs below it."""
    n = len(path_leaf_to_root)
    for i, ca in enumerate(path_leaf_to_root[1:], start=1):
        if ca.path_len is None:
            continue
        below = [
            c for c in path_leaf_to_root[:i]
            if c.subject_der != c.issuer_der and c.is_ca
        ]
        if len(below) > ca.path_len:
            return False, {"rule": "PATH_LEN", "at": ca.fingerprint,
                           "path_len": ca.path_len, "counted_below": len(below)}
    return True, None


def check_eku(path_leaf_to_root: list[ParsedCert]) -> tuple[bool, dict | None]:
    """Code-signing EKU profile: leaf must assert codeSigning; any EKU in a
    CA must also include it (RFC 5280 §4.2.1.12)."""
    leaf = path_leaf_to_root[0]
    if leaf.eku is None or OID_EKU_CODE_SIGNING not in leaf.eku:
        return False, {"rule": "EKU", "at": leaf.fingerprint,
                       "required": OID_EKU_CODE_SIGNING, "present": sorted(leaf.eku or [])}
    for ca in path_leaf_to_root[1:]:
        if ca.eku is not None and OID_EKU_CODE_SIGNING not in ca.eku:
            return False, {"rule": "EKU", "at": ca.fingerprint,
                           "present": sorted(ca.eku)}
    return True, None


def process_policies(path_leaf_to_root: list[ParsedCert],
                     initial_policies: frozenset[str]) -> tuple[bool, dict | None, list[dict]]:
    """RFC 5280 §6.1.3/§6.1.4 policy processing with an explicit policy tree.

    Covers certificatePolicies intersection, anyPolicy suppression via
    inhibitAnyPolicy / requireExplicitPolicy, and policyMappings that are
    allowed or inhibited layer by layer. Qualifiers (CPS URIs / user notices)
    are accepted as present but never influence the decision.

    The per-certificate transition (:func:`policy_step`) and the §6.1.5
    wrap-up (:func:`policy_wrap_up`) operate on immutable, canonical
    valid-policy-tree levels, so callers (path search) can memoize them and
    merge DER-distinct cross-certificates with identical policy content.
    """
    path = list(reversed(path_leaf_to_root))  # root -> leaf
    n = len(path)
    level = (_PolicyTreeNode(OID_ANY_POLICY, (OID_ANY_POLICY,), ()),)
    # Defaults from RFC 5280 §6.1.2 (b)/(e); user supplies no initial
    # constraint values.
    counters = (n, n, n + 1)  # explicit, inhibitAnyPolicy, inhibitPolicyMapping
    trace: list[dict] = []

    for i, cert in enumerate(path, start=1):
        failure, result = policy_step(cert, i, counters, level)
        if failure is not None:
            return False, {"rule": "POLICY", "at": cert.fingerprint,
                           **failure}, trace
        level, counters = result
        trace.append({
            "certificate": cert.fingerprint,
            "depth": i,
            "certificate_policies": sorted(cert.policies.oids),
            "valid_policies": sorted({k.policy for k in level}),
            "explicit_policy_remaining": counters[0],
            "inhibit_any_policy_remaining": counters[1],
            "inhibit_policy_mapping_remaining": counters[2],
            "applied_mappings": [list(m) for m in cert.policy_mappings],
        })

    ok, detail = policy_wrap_up(level, counters[0], frozenset(initial_policies))
    if not ok:
        return False, {"rule": "POLICY", "at": path[-1].fingerprint, **detail}, trace
    return True, detail, trace


class _PolicyTreeNode:
    """Canonical node of the valid policy tree.

    Two nodes with equal ``(policy, expected, lineage)`` are merged even when
    they have different tree parents: only these three fields influence any
    later matching decision, so the merge preserves the verdict. ``expected``
    is the set of policy OIDs that may still match below the node;
    ``lineage`` accumulates every concrete OID seen or mapped on the
    root→node chain so the final intersection can match mapped policies.
    """

    __slots__ = ("policy", "expected", "lineage")

    def __init__(self, policy: str, expected, lineage=frozenset()):
        self.policy = policy
        self.expected = frozenset(expected)
        self.lineage = frozenset(lineage)

    def __eq__(self, other):
        return (self.policy == other.policy
                and self.expected == other.expected
                and self.lineage == other.lineage)

    def __hash__(self):
        return hash((self.policy, self.expected, self.lineage))

    def __lt__(self, other):
        return (self.policy, tuple(sorted(self.expected)),
                tuple(sorted(self.lineage))) < (
            other.policy, tuple(sorted(other.expected)),
            tuple(sorted(other.lineage)))


def policy_step(cert: ParsedCert, i: int, counters: tuple[int, int, int],
                level: tuple[_PolicyTreeNode, ...]
                ) -> tuple[dict | None, tuple | None]:
    """Process certificate ``i`` (trust anchor = 1) of a root→leaf path.

    ``counters`` is ``(explicit, inhibit_any, inhibit_map)`` before this
    certificate; on success returns ``(None, (new_level, new_counters))``.
    """
    ANY = OID_ANY_POLICY
    explicit, inhibit_any, inhibit_map = counters
    self_issued = cert.subject_der == cert.issuer_der
    mappings = list(cert.policy_mappings)

    # RFC 5280 §6.1.3 ordering: the certificate is matched against the
    # *current* counters; non-self-issued certs then consume one level;
    # the certificate's own constraint extensions finally take effect
    # for certificates below it. This reproduces the PKITS semantics
    # (e.g. inhibitAnyPolicy SkipCerts=1 admits anyPolicy in the first
    # CA but suppresses it in the next).
    if inhibit_map == 0 and mappings:
        return {"reason": "policyMapping_inhibited",
                "mappings": [list(m) for m in mappings]}, None

    cps = sorted(cert.policies.oids)
    kids: list[_PolicyTreeNode] = []
    # (b)/(c) key policy matching.
    for p in cps:
        for node in level:
            if p == ANY:
                if node.policy == ANY and (i == 1 or self_issued
                                           or inhibit_any > 0):
                    kids.append(_PolicyTreeNode(ANY, node.expected,
                                                node.lineage))
                else:
                    # anyPolicy suppressed: expand only concrete expected
                    # policies, never re-introduce ANY.
                    for o in sorted(node.expected):
                        if o == ANY:
                            continue
                        kids.append(_PolicyTreeNode(o, (o,),
                                                    node.lineage | {o}))
            elif node.policy == ANY or p in node.expected:
                kids.append(_PolicyTreeNode(p, (p,), node.lineage | {p}))

    # (d) policy mappings.
    if mappings:
        expected_sets = [set(n.expected) for n in level]
        for idp, sdp in mappings:
            for e in expected_sets:
                if idp in e:
                    e.add(sdp)
                    e.discard(idp)
        for idp, sdp in mappings:
            for pi, node in enumerate(level):
                if node.policy == ANY and sdp in expected_sets[pi]:
                    kids.append(_PolicyTreeNode(sdp, (sdp,),
                                                node.lineage | {idp, sdp}))
            for ki, k in enumerate(kids):
                if k.policy == idp:
                    kids[ki] = _PolicyTreeNode(sdp, (sdp,),
                                               k.lineage | {idp, sdp})

    # anyPolicy suppression at this depth.
    if inhibit_any == 0 and i > 1 and not self_issued:
        kids = [k for k in kids if k.policy != ANY]
    if not kids:
        return {"reason": "valid_policy_tree_empty",
                "certificate_policies": cps}, None

    # Consume one level, then adopt this certificate's constraints.
    if not self_issued:
        explicit = max(explicit - 1, 0)
        inhibit_any = max(inhibit_any - 1, 0)
        inhibit_map = max(inhibit_map - 1, 0)
    if cert.require_explicit_policy is not None:
        explicit = min(explicit, cert.require_explicit_policy)
    if cert.inhibit_policy_mapping is not None:
        inhibit_map = min(inhibit_map, cert.inhibit_policy_mapping)
    if cert.inhibit_any_policy is not None:
        inhibit_any = min(inhibit_any, cert.inhibit_any_policy)

    return None, (tuple(sorted(set(kids))),
                  (explicit, inhibit_any, inhibit_map))


def policy_wrap_up(level: tuple[_PolicyTreeNode, ...], explicit: int,
                   initial_policies: frozenset[str]) -> tuple[bool, dict]:
    """§6.1.5 wrap-up policy intersection over the final tree level."""
    ANY = OID_ANY_POLICY
    authority_concrete: set[str] = set()
    has_any_leaf = False
    for k in level:
        if k.policy == ANY:
            has_any_leaf = True
        else:
            authority_concrete |= k.lineage | {k.policy}
    user = set(initial_policies)
    if user == {ANY}:
        result = set(authority_concrete)
        if has_any_leaf:
            result.add(ANY)
    else:
        result = authority_concrete & user
        # An anyPolicy leaf satisfies any explicitly requested user policy
        # unless explicit policy was required before the leaf.
        if has_any_leaf and explicit > 0:
            result |= user
    if not result:
        return False, {"reason": "no_acceptable_policy",
                       "authority_policies": sorted(
                           {k.policy for k in level}),
                       "initial_policies": sorted(user)}
    return True, {"user_constrained_policy_set": sorted(result)}


def validate_static_chain(path_leaf_to_root: list[ParsedCert],
                          signed_at: int,
                          initial_policies: frozenset[str]) -> tuple[bool, dict | None, list[dict]]:
    """Validity windows, basic constraints/KU are checked during graph
    traversal; this re-runs name/pathLen/EKU/policy on the complete chain
    and returns the policy trace for the evidence package."""
    ok, detail = check_path_len(path_leaf_to_root)
    if not ok:
        return False, detail, []
    ok, detail = check_name_constraints(path_leaf_to_root)
    if not ok:
        return False, detail, []
    ok, detail = check_eku(path_leaf_to_root)
    if not ok:
        return False, detail, []
    ok, detail, trace = process_policies(path_leaf_to_root, initial_policies)
    if not ok:
        return False, detail, trace
    return True, detail, trace
