import { Spinner } from "./Spinner";

export function PageLoader() {
  return (
    <div className="flex min-h-[60vh] w-full items-center justify-center">
      <Spinner className="h-8 w-8" />
    </div>
  );
}
