export type GroupMode = "precision" | "balanced" | "recall";

export interface PredictionResponse {
  prob_hate: number;
  prob_nonhate: number;
  pred: number;
  label: string;
}

export interface SettingsResponse {
  chat_id: number;
  enabled: boolean;
  threshold: number;
  mode: GroupMode;
  retention_days: number;
  updated_at: string;
}

export interface StatsResponse {
  chat_id: number;
  window: string;
  total_events: number;
  blocked: number;
  warned: number;
  deleted: number;
  top_offenders: string[];
}

export interface EventPayload {
  chat_id: number;
  user_id?: number;
  username?: string;
  message_id?: number;
  prob_hate: number;
  prob_nonhate: number;
  action: string;
  text?: string;
  reason?: string;
}

export interface GroupSummary {
  chat_id: number;
  enabled: boolean;
  threshold: number;
  mode: GroupMode;
  updated_at: string;
  title?: string;
  group_type?: string;
  last_active: string;
}

export interface AdminEntry {
  id: number;
  chat_id: number;
  user_id: number;
  added_at: string;
}

export type MemberStatus = "muted" | "banned";

export interface MemberModeration {
  id: number;
  chat_id: number;
  user_id: number;
  username?: string;
  status: MemberStatus;
  reason?: string;
  expires_at?: string;
  created_at: string;
}
