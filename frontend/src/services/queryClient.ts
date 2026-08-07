import { QueryClient } from "@tanstack/react-query";

/**
 * All server state (predictions, history) flows through React Query, not
 * Context — it needs caching, background refetch, and loading/error states
 * that Context would otherwise require reinventing by hand.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 0, // predictions are not silently retried — see apiClient.ts
    },
  },
});
