import { FaAutoprefixer, FaHome, FaRegQuestionCircle } from "react-icons/fa";
import { TbNumbers, TbVocabulary } from "react-icons/tb";
import Logo from "../assets/images/logo.png";
import { H1, Typography } from "./typography";
import { BsAlphabet } from "react-icons/bs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useRef, useState } from "react";
import useMobile from "@/hooks/useMobile";

const navData = [
  {
    name: "Home",
    href: "/",
    icon: <FaHome />,
  },
  {
    name: "Tentang",
    href: "/tentang",
    icon: <FaRegQuestionCircle />,
  },
  {
    name: "Kosakata",
    href: "/kosakata",
    icon: <TbVocabulary />,
    sections: [
      {
        name: "Abjad",
        href: "?tab=abjad",
        icon: <BsAlphabet />,
      },
      {
        name: "Imbuhan",
        href: "?tab=imbuhan",
        icon: <FaAutoprefixer />,
      },
      {
        name: "Angka",
        href: "?tab=angka",
        icon: <TbNumbers />,
      },
    ],
  },
  {
    name: "Pencarian",
    href: "/search",
    icon: <TbVocabulary />,
    sections: [
      {
        name: "Kata",
        href: "/kata",
        icon: <BsAlphabet />,
      },
      {
        name: "Kalimat",
        href: "/kalimat",
        icon: <FaAutoprefixer />,
      },
    ],
  },
];

const NavLink = ({
  href,
  icon,
  name,
  isChild,
  showText,
}: {
  href: string;
  icon: JSX.Element;
  name: string;
  isChild?: boolean;
  showText?: boolean;
}) => (
  <a
    href={href}
    className={cn(
      "flex items-center gap-4 py-2 rounded-2xl font-bold",
      !isChild ? "hover:bg-[#333]" : "hover:bg-[#808080] ml-4 my-1",
      showText ? "px-12" : "px-2 justify-center ml-0"
    )}
  >
    {icon}
    <Typography
      className={cn(showText ? "text-white font-bold" : "hidden")}
      level={isChild ? "base" : "lg"}
    >
      {name}
    </Typography>
  </a>
);

export default function Navbar() {
  const isMobile = useMobile();
  const [showText, setShowText] = useState(isMobile ? false : true);
  const navRef = useRef<HTMLDivElement>(null);

  const handleClickOutside = useCallback(
    (event: MouseEvent) => {
      if (
        navRef.current &&
        !navRef.current.contains(event.target as Node) &&
        isMobile
      ) {
        setShowText(false);
      }
    },
    [isMobile]
  );

  useEffect(() => {
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [handleClickOutside]);

  return (
    <>
      <nav
        ref={navRef}
        className={cn(
          "bg-[#000] shadow-md h-screen",
          isMobile && showText && "fixed z-10"
        )}
      >
        <div className="flex flex-col gap-3 py-7 px-3">
          {/* Icon apps */}
          <button
            className="flex gap-4 items-center self-center"
            onClick={() => setShowText((prev) => !prev)}
          >
            <img src={Logo} width={40} height={40} alt="Logo" />
            <H1
              className={cn(showText ? "text-white font-black" : "hidden")}
              level="xl"
            >
              Apps
            </H1>
          </button>

          <span className="bg-slate-700 h-[1px] w-full" />

          {/* Navdata link mapping */}
          <ul>
            {navData.map((navItem, index) => (
              <li key={index} className="py-2">
                {navItem.sections && navItem.sections.length > 0 ? (
                  // Multiple child url
                  <Accordion
                    type="single"
                    collapsible
                    className="rounded-2xl hover:bg-[#333]"
                  >
                    <AccordionItem value={`item-${index}`}>
                      <AccordionTrigger
                        isShow={showText}
                        className={cn(
                          "flex items-center gap-4",
                          showText ? "px-12" : "px-2 justify-center"
                        )}
                      >
                        {navItem.icon}
                        <Typography
                          className={cn(
                            showText ? "text-white font-bold" : "hidden"
                          )}
                          level="lg"
                        >
                          {navItem.name}
                        </Typography>
                      </AccordionTrigger>
                      <AccordionContent>
                        <ul>
                          {navItem.sections.map((section, sectionIndex) => (
                            <li key={sectionIndex}>
                              <NavLink
                                href={navItem.href + section.href}
                                icon={section.icon}
                                name={section.name}
                                showText={showText}
                                isChild
                              />
                            </li>
                          ))}
                        </ul>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                ) : (
                  // Single Url
                  <NavLink
                    href={navItem.href}
                    showText={showText}
                    icon={navItem.icon}
                    name={navItem.name}
                  />
                )}
              </li>
            ))}
          </ul>
        </div>

        {/* Overlay */}
      </nav>
      {showText && isMobile && (
        <div className="bg-black -z-[5] h-screen w-screen bg-opacity-40 absolute" />
      )}
    </>
  );
}
