import { H3 } from "./typography";

const Card = ({ image, name }: { image: string; name: string }) => {
  if (!name) return;
  return (
    <div className="rounded-2xl overflow-hidden">
      <img
        src={image}
        alt={name}
        width={250}
        height={250}
        className="object-cover max-sm:max-h-[220px] object-center aspect-square w-full"
      />
      <H3
        className="bg-slate-800 p-2 text-white font-medium text-center"
        level={"xl"}
      >
        {name}
      </H3>
    </div>
  );
};

export default Card;
