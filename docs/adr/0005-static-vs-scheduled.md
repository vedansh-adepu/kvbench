# 0005 — Separate static and scheduled answers

Static output is a phase-paged worst case for specified concurrency/context;
scheduled output uses actual arrivals and admission. Never take the maximum
of static/scheduled peaks, which would erase arrival sensitivity. Both use the
same memory/budget functions. Trade-off: users must choose which scenario they
are asking about instead of receiving one ambiguous memory number.
