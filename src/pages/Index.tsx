// src/pages/Index.tsx (DROP-IN PATCH)
// Only replace the marked blocks below. Everything else can stay as-is.

// 1) Replace your withCacheBuster with this simpler version (no URL() needed)
const withCacheBuster = (url: string, v: string | number) =>
  `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(String(v))}`;

// 2) Replace your entire useEffect(...) with this version
useEffect(() => {
  let alive = true;

  const load = async () => {
    try {
      let cacheKey = Date.now();

      try {
        const lastRunRes = await fetch(
          withCacheBuster(`${dataBaseUrl}last_run.json`, Date.now()),
        );

        if (lastRunRes.ok) {
          const lastRunJson = (await lastRunRes.json()) as {
            last_run?: string;
            ts?: string | number;
          };

          const lastRunRaw = lastRunJson?.ts ?? lastRunJson?.last_run;

          if (lastRunRaw != null) {
            const numeric = Number(lastRunRaw);

            if (!Number.isNaN(numeric)) {
              // seconds -> ms, sonst ist es schon ms
              cacheKey = numeric < 10_000_000_000 ? numeric * 1000 : numeric;
            } else {
              const parsed = Date.parse(String(lastRunRaw));
              if (!Number.isNaN(parsed)) {
                cacheKey = parsed;
              }
            }
          }
        }
      } catch (err) {
        console.warn("Failed to load last_run.json for cache busting.", err);
      }

      const [payloadRes, stateRes] = await Promise.all([
        fetch(withCacheBuster(`${dataBaseUrl}dashboard_payload.json`, cacheKey)),
        fetch(withCacheBuster(`${dataBaseUrl}dashboard_state.json`, cacheKey)),
      ]);

      if (!payloadRes.ok || !stateRes.ok) {
        throw new Error("Failed to load dashboard data.");
      }

      const payloadJson = (await payloadRes.json()) as DashboardPayload;
      const stateJson = (await stateRes.json()) as DashboardState;

      if (alive) {
        setPayload(payloadJson);
        setDashboardState(stateJson);
        setLoadError(null); // ✅ clear stale error on success
      }
    } catch (err) {
      if (alive) {
        setLoadError(err instanceof Error ? err.message : "Failed to load data.");
      }
    }
  };

  load();

  return () => {
    alive = false;
  };
}, [dataBaseUrl]);

// 3) In your ECE StatCard, replace ONLY the subtitle line with this:
// subtitle="Before: —"
