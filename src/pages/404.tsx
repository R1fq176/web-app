// NotFound.jsx or NotFound.tsx

import { H2, H3 } from "@/components/typography";
import Metadata from "@/lib/metadata";

const NotFound = () => {
  return (
    <main className="not-found text-black flex flex-col gap-5 w-full self-center text-center">
      <Metadata title="404 | Sign App" />

      <H2 level={"5xl"} className="font-black">
        404 - Page Not Found
      </H2>
      <H3 level={"2xl"}>The page you are looking for does not exist.</H3>
    </main>
  );
};

export default NotFound;
