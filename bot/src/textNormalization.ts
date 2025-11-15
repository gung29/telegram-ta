// Utilities to normalize obfuscated text (leet, repeated chars) before evaluation.
const RISKY_TERMS = new Set([
  "anjing",
  "bangsat",
  "babi",
  "bodoh",
  "brengsek",
  "goblok",
  "hina",
  "kafir",
  "tolol",
]);

const LEETSPEAK_MAP: Record<string, string> = {
  "0": "o",
  "1": "i",
  "!": "i",
  "3": "e",
  "4": "a",
  "@": "a",
  "5": "s",
  "$": "s",
  "6": "g",
  "7": "t",
  "8": "b",
  "9": "g",
};

const TOKEN_CHUNK_RE = /[A-Za-z0-9*@#\$%&!?\-_/]+/g;
const NON_WORD_RE = /[^a-z0-9]/g;
const REPEATED_CHAR_RE = /(.)\1{2,}/g;
const OBFUSCATION_SIMILARITY = 0.8;

const applyLeetMap = (token: string): string =>
  token
    .toLowerCase()
    .split("")
    .map((ch) => LEETSPEAK_MAP[ch] ?? ch)
    .join("");

const cleanCandidate = (token: string): string => {
  const normalized = applyLeetMap(token).replace(NON_WORD_RE, "");
  return normalized.replace(REPEATED_CHAR_RE, "$1$1");
};

const levenshteinDistance = (a: string, b: string): number => {
  if (a === b) return 0;
  const rows = a.length + 1;
  const cols = b.length + 1;
  const dp = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
  for (let i = 0; i < rows; i += 1) {
    dp[i][0] = i;
  }
  for (let j = 0; j < cols; j += 1) {
    dp[0][j] = j;
  }
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[a.length][b.length];
};

const similarityRatio = (a: string, b: string): number => {
  if (!a.length && !b.length) return 1;
  if (!a.length || !b.length) return 0;
  const distance = levenshteinDistance(a, b);
  const maxLen = Math.max(a.length, b.length, 1);
  return 1 - distance / maxLen;
};

const matchRiskyWord = (candidate: string): string | null => {
  if (!candidate) return null;
  if (RISKY_TERMS.has(candidate)) return candidate;
  for (const risky of RISKY_TERMS) {
    const maxLen = Math.max(candidate.length, risky.length);
    if (maxLen < 4) continue;
    if (similarityRatio(candidate, risky) >= OBFUSCATION_SIMILARITY) {
      return risky;
    }
  }
  return null;
};

export const normalizeObfuscatedTerms = (text: string): { normalized: string; flagged: string[] } => {
  const flagged = new Set<string>();
  const normalized = text.replace(TOKEN_CHUNK_RE, (token) => {
    const canonical = matchRiskyWord(cleanCandidate(token));
    if (canonical) {
      flagged.add(canonical);
      return canonical;
    }
    return token;
  });
  return {
    normalized,
    flagged: Array.from(flagged).sort(),
  };
};
