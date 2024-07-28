import Card from "@/components/card";
import { H2, P, Typography } from "@/components/typography";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { instructionWordPage } from "@/constants/content/kata.page";
import { kataDasarData } from "@/constants/data/kata";
import Metadata from "@/lib/metadata";
import { CardProps } from "@/types/general";
import { useState } from "react";

export default function KataPage() {
  // State for search input, loading status, search status, and search results
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSearch, setIsSearch] = useState(false);
  const [result, setResult] = useState<CardProps[]>([]);

  // Function to handle search logic
  const handleSearch = () => {
    if (!search) {
      handleClear();
      return;
    }
    try {
      setLoading(true); // Set loading to true when starting search
      const res = kataDasarData.filter((item) =>
        item.name.toLowerCase().includes(search.toLowerCase())
      );

      setResult(res); // Update search results
      setIsSearch(true); // Set search status to true
    } catch (e) {
      console.error(e); // Log errors if any occur
    } finally {
      setLoading(false); // Set loading to false after search is complete
    }
  };

  // Function to clear search results and input
  const handleClear = () => {
    setIsSearch(false); // Reset search status
    setResult([]); // Clear search results
    setSearch(""); // Clear search input
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <main className="bg-white w-full overflow-y-auto text-black flex flex-col p-4 lg:p-10 xl:p-12 gap-6 lg:gap-12">
      <Metadata title="Kata | Sign App" />

      {/* Page title */}
      <H2 className="font-bold" level={"3xl"}>
        Pencarian Kata
      </H2>
      <section className="flex flex-col gap-7">
        {/* Search and clear buttons */}
        <div className="flex items-center gap-2">
          <Input
            value={search}
            disabled={loading}
            placeholder="Masukkan kata dasar yang ingin Anda cari"
            onKeyDown={handleKeyPress}
            onChange={(e) => setSearch(e.target.value)} // Update search input
          />
          <Button onClick={handleClear} disabled={loading}>
            Clear
          </Button>
          <Button onClick={handleSearch} disabled={loading}>
            Cari
          </Button>
        </div>

        {/* Content area: either search results or guidance */}
        <div>
          <div className="py-3 px-5 bg-slate-800 rounded-t-3xl">
            <H2 level={"lg"} className="text-white font-bold">
              {isSearch ? "Hasil Pencarian" : "Panduan"}
            </H2>
          </div>
          <div className="border-2 border-slate-800 rounded-b-3xl p-6 space-y-5">
            {!loading ? (
              isSearch ? (
                result.length > 0 ? (
                  // Display search results in a grid layout
                  <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 2xl:grid-cols-6 gap-6">
                    {result.map((item) => (
                      <Card
                        key={item.name}
                        image={item.image}
                        name={item.name}
                      />
                    ))}
                  </div>
                ) : (
                  <P className="text-center font-semibold" level={"lg"}>
                    Kata yang anda cari tidak berhasil ditemukan
                  </P>
                )
              ) : (
                // Display guidance instructions
                <ul className="flex flex-col gap-2">
                  {instructionWordPage.map((item, index) => (
                    <li key={index}>
                      <Typography level={"lg"} className="font-medium">
                        {item}
                      </Typography>
                    </li>
                  ))}
                </ul>
              )
            ) : (
              <P level={"lg"} className="font-semibold">
                Loading....
              </P>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
