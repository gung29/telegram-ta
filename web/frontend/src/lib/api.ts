import { withHeaders } from "./telegram";

export class HttpError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

const baseFetch = async <T>(input: RequestInfo, init?: RequestInit): Promise<T> => {
  const response = await fetch(input, {
    ...init,
    headers: {
      ...withHeaders(),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      /* ignore */
    }
    throw new HttpError(detail, response.status);
  }

  if (response.status === 204) {
    return null as T;
  }

  return (await response.json()) as T;
};

export interface SettingsResponse {
  chat_id: number;
  enabled: boolean;
  threshold: number;
  mode: string;
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

export interface TestResponse {
  prob_hate: number;
  prob_nonhate: number;
  label: string;
}

export interface GroupSummary {
  chat_id: number;
  enabled: boolean;
  threshold: number;
  mode: string;
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

export interface ActivityPoint {
  date: string;
  deleted: number;
  warned: number;
  blocked: number;
}

export interface ActivityResponse {
  chat_id: number;
  points: ActivityPoint[];
}

export interface EventEntry {
  id: number;
  chat_id: number;
  user_id?: number;
  username?: string;
  prob_hate: number;
  prob_nonhate: number;
  action: string;
  text?: string;
  reason?: string;
  created_at: string;
  manual_label?: string | null;
  manual_verified: boolean;
  manual_verified_at?: string | null;
}

export interface UserActionSummary {
  user_id: number;
  username?: string;
  warnings_today: number;
  mutes_total: number;
  last_warning?: string | null;
  last_mute?: string | null;
}

const qs = (params: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();

export const fetchSettings = (chatId: number) =>
  baseFetch<SettingsResponse>(`/api/settings?${qs({ chat_id: chatId })}`);

export const updateSettings = (chatId: number, payload: Partial<SettingsResponse>) =>
  baseFetch<SettingsResponse>(`/api/settings?${qs({ chat_id: chatId })}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const fetchStats = (chatId: number, window: string) =>
  baseFetch<StatsResponse>(`/api/stats?${qs({ chat_id: chatId, window })}`);

export const fetchGroups = () => baseFetch<GroupSummary[]>("/api/groups");

export const fetchAdmins = (chatId: number) =>
  baseFetch<AdminEntry[]>(`/api/admins?${qs({ chat_id: chatId })}`);

export const addAdmin = (chatId: number, userId: number) =>
  baseFetch(`/api/admins?${qs({ chat_id: chatId })}`, {
    method: "POST",
    body: JSON.stringify({ user_id: userId }),
  });

export const removeAdmin = (chatId: number, userId: number) =>
  baseFetch(`/api/admins/${userId}?${qs({ chat_id: chatId })}`, {
    method: "DELETE",
  });

export const fetchMembers = (chatId: number, status?: MemberStatus) =>
  baseFetch<MemberModeration[]>(`/api/members?${qs({ chat_id: chatId, ...(status ? { status } : {}) })}`);

export const createMemberStatus = (
  chatId: number,
  payload: { user_id: number; username?: string; status: MemberStatus; duration_minutes?: number; reason?: string },
) =>
  baseFetch(`/api/members?${qs({ chat_id: chatId })}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const deleteMemberStatus = (chatId: number, userId: number, status: MemberStatus) =>
  baseFetch(`/api/members/${userId}?${qs({ chat_id: chatId, status })}`, {
    method: "DELETE",
  });

export const fetchActivity = (chatId: number, days = 7) =>
  baseFetch<ActivityResponse>(`/api/activity?${qs({ chat_id: chatId, days })}`);

export const fetchEvents = (chatId: number, limit = 50, offset = 0) =>
  baseFetch<EventEntry[]>(`/api/events?${qs({ chat_id: chatId, limit, offset })}`);

export const fetchEventCount = async (chatId: number) => {
  const data = await baseFetch<{ total: number }>(`/api/events/count?${qs({ chat_id: chatId })}`);
  return data.total ?? 0;
};

export const verifyEvent = (chatId: number, eventId: number, label: "hate" | "non-hate") =>
  baseFetch<EventEntry>(`/api/events/${eventId}/verify?${qs({ chat_id: chatId })}`, {
    method: "POST",
    body: JSON.stringify({ label }),
  });

export const runTest = (chatId: number, text: string) =>
  baseFetch<TestResponse>(`/api/test?${qs({ chat_id: chatId })}`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });

export const fetchUserActions = (chatId: number) =>
  baseFetch<UserActionSummary[]>(`/api/user_actions?${qs({ chat_id: chatId })}`);

export const resetUserAction = (chatId: number, userId: number, action: "warned" | "muted") =>
  baseFetch(`/api/user_actions/${userId}/reset?${qs({ chat_id: chatId })}`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });

export const exportCsv = async (chatId: number) => {
  const response = await fetch(`/api/export?${qs({ chat_id: chatId })}`, {
    headers: withHeaders(),
  });
  if (!response.ok) {
    throw new Error("Gagal mengunduh CSV");
  }
  const blob = await response.blob();
  return blob;
};
