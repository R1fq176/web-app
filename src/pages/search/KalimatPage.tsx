import { useState } from "react";
import Card from "@/components/card";
import { H2, P, Typography } from "@/components/typography";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { imbuhanData } from "@/constants/data/imbuhan";
import { kataDasarData } from "@/constants/data/kata";
import { instructionSentence } from "@/constants/content/kalimat-.page";
import { MatchedResult, matchWord } from "@/lib/search-sentence.calculation";
import Metadata from "@/lib/metadata";

export default function KataPage() {
  const [search, setSearch] = useState<string>(""); // Search input value
  const [loading, setLoading] = useState<boolean>(false); // Loading state
  const [isSearch, setIsSearch] = useState<boolean>(false); // Search state
  const [result, setResult] = useState<MatchedResult[]>([]); // Search result

  // Handle search logic
  const handleSearch = () => {
    if (!search) {
      handleClear();
      return;
    }
    setLoading(true);
    setResult([]);

    const words = search.toLowerCase().trim().split(" ");
    const matchedResults: MatchedResult[] = [];

    words.forEach((word) => {
      const { baseWord, prefix, suffix } = matchWord(
        word,
        kataDasarData,
        imbuhanData.filter((imb) => imb.name.endsWith("-")),
        imbuhanData.filter((imb) => imb.name.startsWith("-"))
      );

      if (baseWord) {
        matchedResults.push({ baseWord, prefix, suffix });
      }
    });

    setResult(matchedResults);
    setIsSearch(true);
    setLoading(false);
  };

  // Clear search input and results
  const handleClear = () => {
    setSearch("");
    setIsSearch(false);
    setResult([]);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <main className="bg-white w-full overflow-y-auto text-black flex flex-col p-4 lg:p-10 xl:p-12 gap-6 lg:gap-12">
      <Metadata title="Kalimat | Sign App" />

      {/* Page title */}
      <H2 className="font-bold" level={"3xl"}>
        Pencarian Kalimat
      </H2>
      <section className="flex flex-col gap-7">
        {/* Search input and buttons */}
        <div className="flex items-center gap-2">
          <Input
            value={search}
            disabled={loading}
            onKeyDown={handleKeyPress}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Masukkan Kalimat yang ingin dicari"
          />
          <Button onClick={handleClear} disabled={loading}>
            Clear
          </Button>
          <Button onClick={handleSearch} disabled={loading}>
            Cari
          </Button>
        </div>

        {/* Instructions or search results */}
        <div>
          <div className="py-3 px-5 bg-slate-800 rounded-t-3xl">
            <H2 level={"lg"} className="text-white font-bold">
              {isSearch ? "Hasil Pencarian" : "Panduan"}
            </H2>
          </div>
          <div className="border-2 border-slate-800 rounded-b-3xl p-6 space-y-5">
            {isSearch ? (
              result && result.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 2xl:grid-cols-6 gap-6">
                  {result.map((item, index) =>
                    item.baseWord ? (
                      <>
                        {item.prefix && (
                          <Card
                            key={`prefix-${index}`}
                            image={item.prefix.image}
                            name={item.prefix.name}
                          />
                        )}
                        <Card
                          key={`base-${index}`}
                          image={item.baseWord.image}
                          name={item.baseWord.name}
                        />
                        {item.suffix && (
                          <Card
                            key={`suffix-${index}`}
                            image={item.suffix.image}
                            name={item.suffix.name}
                          />
                        )}
                      </>
                    ) : null
                  )}
                </div>
              ) : (
                <P className="text-center font-semibold" level={"lg"}>
                  Kalimat yang anda cari tidak berhasil ditemukan
                </P>
              )
            ) : (
              <ul className="flex flex-col gap-2">
                {instructionSentence.map((item, index) => (
                  <li key={index}>
                    <Typography level={"lg"} className="font-medium">
                      {item}
                    </Typography>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
