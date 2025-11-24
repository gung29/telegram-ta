export enum View {
  DASHBOARD = 'DASHBOARD',
  STATS = 'STATS',
  LOGS = 'LOGS',
  VERIFY = 'VERIFY',
  ADMIN = 'ADMIN'
}

export interface ModerationLog {
  id: string;
  timestamp: string;
  user: string;
  userId: string;
  action: 'Muted' | 'Banned' | 'Flagged' | 'Deleted' | 'Warned';
  hateScore: number;
  content: string;
  status: 'Auto-Muted' | 'Manual Review' | 'Verified' | 'Active';
}

export interface VerificationItem {
  id: string;
  user: string;
  timeAgo: string;
  content: string;
  score: number;
  avatarId: number;
}

export interface UserOffender {
  id: string;
  username: string;
  userId: string;
  warnings: number;
  mutes: number;
  lastActivity: string;
  avatarId: number;
}