import Camera from "@/components/camera";
import { H2, P } from "@/components/typography";
import Metadata from "@/lib/metadata";
export default function Home() {
  return (
    <main className="bg-white w-full overflow-y-auto text-black flex flex-col p-4 lg:p-10 xl:p-12 gap-6 lg:gap-12">
      <Metadata title="Study | Sign App" />
      <H2 className="font-bold" level={"3xl"}>
        Study
      </H2>
      {/* Camra comp */}
      <Camera />
      {/* Result appear */}
      <div className="bg-black w-[70%] h-[200px] rounded-xl flex self-center">
        <P
          className="text-white text-center font-medium w-full m-auto"
          level={"xl"}
        >
          Result{" "}
        </P>
      </div>
    </main>
  );
}
