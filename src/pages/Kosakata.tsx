import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { H2, H4, H5 } from "@/components/typography";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import Card from "@/components/card";
import { abjadData } from "@/constants/data/abjad";
import { imbuhanData } from "@/constants/data/imbuhan";
import { angkaData } from "@/constants/data/angka";
import Metadata from "@/lib/metadata";

// Custom hook to parse query parameters from URL
const useQuery = () => {
  return new URLSearchParams(useLocation().search);
};

export default function KosakataPage() {
  const navigate = useNavigate();
  const query = useQuery();
  const tab = query.get("tab") || "abjad"; // Default to "abjad" if no tab is specified

  const [currentPage, setCurrentPage] = useState(1); // Current page for pagination
  const itemsPerPage = 18; // Number of items per page

  // Handle tab change and update URL
  const handleTabChange = useCallback(
    (value: string) => {
      navigate(`/kosakata?tab=${value}`, { replace: true });
    },
    [navigate]
  );

  // Update the tab when query parameter changes
  useEffect(() => {
    handleTabChange(tab);
  }, [tab, handleTabChange]);

  // Paginate data for "angka" tab
  const paginatedAngkaData = angkaData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Check if next/previous page is available
  const isNextAvail =
    currentPage + 1 <= Math.ceil(angkaData.length / itemsPerPage);
  const isPrevAvail = currentPage - 1 >= 1;

  // Handle page changes for pagination
  const handlePageChange = (page: number) => {
    if (page > currentPage && !isNextAvail) return;
    if (page < currentPage && !isPrevAvail) return;
    setCurrentPage(page);
  };

  // Filter imbuhan data for prefixes and suffixes
  const suffixData = imbuhanData.filter((item) => item.name.startsWith("-"));
  const prefixData = imbuhanData.filter((item) => item.name.endsWith("-"));

  return (
    <main className="bg-white w-full overflow-y-auto text-black flex flex-col p-4 lg:p-10 xl:p-12 gap-6 lg:gap-12">
      <Metadata title="kosakata | Sign App" />

      {/* Page title */}
      <H2 className="font-bold" level={"3xl"}>
        Kosakata
      </H2>
      <section className="flex flex-col gap-1">
        {/* Tabs for different categories */}
        <Tabs value={tab} onValueChange={handleTabChange} className="w-full">
          <div className="bg-slate-800 py-3 px-5 rounded-t-3xl">
            <TabsList className="grid w-fit grid-cols-3 py-1 bg-transparent">
              <TabsTrigger
                value="abjad"
                className="px-4 py-1.5 text-slate-400 rounded-lg"
              >
                Abjad
              </TabsTrigger>
              <TabsTrigger
                value="imbuhan"
                className="px-4 py-1.5 text-slate-400 rounded-lg"
              >
                Imbuhan
              </TabsTrigger>
              <TabsTrigger
                value="angka"
                className="px-4 py-1.5 text-slate-400 rounded-lg"
              >
                Angka
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Abjad Content */}
          <TabsContent
            className="border-2 border-slate-800 rounded-b-3xl py-6 px-6 space-y-5"
            value="abjad"
          >
            <H4 className="font-bold text-center" level={"2xl"}>
              Daftar Abjad
            </H4>
            <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 2xl:grid-cols-6 gap-6">
              {abjadData.map((item) => (
                <Card key={item.name} image={item.image} name={item.name} />
              ))}
            </div>
          </TabsContent>

          {/* Imbuhan Content */}
          <TabsContent
            className="border-2 border-slate-800 rounded-b-3xl py-6 px-6 space-y-5"
            value="imbuhan"
          >
            <H4 className="font-bold text-center" level={"2xl"}>
              Daftar Imbuhan
            </H4>
            <div className="space-y-5">
              <H5 level={"xl"} className="font-semibold">
                Awalan
              </H5>
              <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 2xl:grid-cols-6 gap-6">
                {prefixData.map((item) => (
                  <Card key={item.name} image={item.image} name={item.name} />
                ))}
              </div>
              <H5 level={"xl"} className="font-semibold">
                Akhiran
              </H5>
              <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 2xl:grid-cols-6 gap-6">
                {suffixData.map((item) => (
                  <Card key={item.name} image={item.image} name={item.name} />
                ))}
              </div>
            </div>
          </TabsContent>

          {/* Angka Content */}
          <TabsContent
            className="border-2 border-slate-800 rounded-b-3xl py-6 px-6 space-y-5"
            value="angka"
          >
            <H4 className="font-bold text-center" level={"2xl"}>
              Daftar Angka
            </H4>
            <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 2xl:grid-cols-6 gap-6">
              {paginatedAngkaData.map((item) => (
                <Card key={item.name} image={item.image} name={item.name} />
              ))}
            </div>
            {/* Pagination controls */}
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    className={!isPrevAvail ? "cursor-not-allowed" : ""}
                    onClick={() => handlePageChange(currentPage - 1)}
                  />
                </PaginationItem>
                {Array.from(
                  { length: Math.ceil(angkaData.length / itemsPerPage) },
                  (_, index) => (
                    <PaginationItem key={index + 1}>
                      <PaginationLink
                        onClick={() => handlePageChange(index + 1)}
                        isActive={currentPage === index + 1}
                      >
                        {index + 1}
                      </PaginationLink>
                    </PaginationItem>
                  )
                )}
                <PaginationItem>
                  <PaginationNext
                    className={!isNextAvail ? "cursor-not-allowed" : ""}
                    onClick={() => handlePageChange(currentPage + 1)}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </TabsContent>
        </Tabs>
      </section>
    </main>
  );
}
