# Contributing

The list is a path, not a pile. A new resource has to make the path clearer or fill a gap.

## Before opening a pull request

Answer four questions:

1. What mechanism does this source explain?
2. Why is it a primary source?
3. Where does it belong in the order?
4. Which existing source does it replace, if the section already has eight links?

Primary means the paper that introduced the mechanism, the specification or official documentation, the implementing repository, or an operator or implementer report with hardware, measurements, and enough detail to reproduce it.

Do not submit summaries, generic tutorials, vendor marketing, broad surveys, or anything built on software RDMA emulation or simulators.

## Performance claims

A number needs the NIC and firmware, switch, topology, message sizes, software versions, and baseline. If any is missing, omit the number.

## Frontier entries

Frontier items are dated. They move into the core list once a specification or paper, a shipped implementation, and reproducible measurements exist.

## Checks

Run `python3 scripts/check_links.py` before opening the pull request.
