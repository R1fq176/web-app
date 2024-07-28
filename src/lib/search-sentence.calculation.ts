import type { CardProps } from "@/types/general";

// Mismatch calculation function based on levenstein algorithm
const levenshteinDistance = (a: string, b: string): number => {
  const matrix = Array.from({ length: b.length + 1 }, () =>
    Array(a.length + 1).fill(0)
  );

  for (let i = 0; i <= a.length; i++) {
    matrix[0][i] = i;
  }

  for (let j = 0; j <= b.length; j++) {
    matrix[j][0] = j;
  }

  for (let j = 1; j <= b.length; j++) {
    for (let i = 1; i <= a.length; i++) {
      if (b[j - 1] === a[i - 1]) {
        matrix[j][i] = matrix[j - 1][i - 1];
      } else {
        matrix[j][i] =
          Math.min(matrix[j - 1][i - 1], matrix[j][i - 1], matrix[j - 1][i]) +
          1;
      }
    }
  }

  return matrix[b.length][a.length];
};

// Function to remove prefix and suffix from a word
const removeAffix = (
  word: string,
  prefixes: CardProps[],
  suffixes: CardProps[]
): string => {
  let modifiedWord = word;

  // Remove prefix
  prefixes.forEach((prefix) => {
    const prefixStr = prefix.name.slice(0, -1).toLowerCase();
    if (modifiedWord.startsWith(prefixStr)) {
      modifiedWord = modifiedWord.slice(prefixStr.length);
    }
  });

  // Remove suffix
  suffixes.forEach((suffix) => {
    const suffixStr = suffix.name.slice(1).toLowerCase();
    if (modifiedWord.endsWith(suffixStr)) {
      modifiedWord = modifiedWord.slice(0, -suffixStr.length);
    }
  });

  return modifiedWord;
};

// Function to check if a word matches a pattern with a base word and affixes
const matchWord = (
  word: string,
  kataDasar: CardProps[],
  imbuhanPrefix: CardProps[],
  imbuhanSuffix: CardProps[]
): MatchedResult => {
  let baseWord: CardProps | null = null;
  let prefix: CardProps | null = null;
  let suffix: CardProps | null = null;
  let minDistance = Infinity;

  const modifiedWord = removeAffix(word, imbuhanPrefix, imbuhanSuffix);

  kataDasar.forEach((kd) => {
    const distance = levenshteinDistance(modifiedWord, kd.name.toLowerCase());
    if (distance < minDistance && distance <= 2) {
      // Only consider base words with ≤ 2 mismatches
      minDistance = distance;
      baseWord = kd;
    }
  });

  if (baseWord) {
    imbuhanPrefix.forEach((imp) => {
      const imbuhanPref = imp.name.toLowerCase().slice(0, -1);
      if (
        !baseWord?.name.startsWith(imbuhanPref) &&
        word.startsWith(imbuhanPref)
      ) {
        prefix = imp;
      }
    });

    imbuhanSuffix.forEach((ims) => {
      const imbuhanSuff = ims.name.toLowerCase().slice(1);
      if (!baseWord?.name.endsWith(imbuhanSuff) && word.endsWith(imbuhanSuff)) {
        suffix = ims;
      }
    });
  }

  return { baseWord, prefix, suffix };
};

interface MatchedResult {
  baseWord: CardProps | null;
  prefix: CardProps | null;
  suffix: CardProps | null;
}

export { levenshteinDistance, removeAffix, matchWord, type MatchedResult };
