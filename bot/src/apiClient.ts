import axios from "axios";
import { botConfig } from "./config";
import {
  AdminEntry,
  EventPayload,
  GroupSummary,
  MemberModeration,
  MemberStatus,
  PredictionResponse,
  SettingsResponse,
  StatsResponse,
} from "./types";

const client = axios.create({
  baseURL: botConfig.apiUrl,
  timeout: 10000,
  headers: {
    "X-API-Key": botConfig.apiKey,
  },
});

export const predictText = async (text: string): Promise<PredictionResponse> => {
  const { data } = await client.post<PredictionResponse>("/predict", { text });
  return data;
};

export const fetchSettings = async (chatId: number): Promise<SettingsResponse> => {
  const { data } = await client.get<SettingsResponse>(`/admin/settings/${chatId}`);
  return data;
};

export const sendSettings = async (chatId: number, payload: Partial<SettingsResponse>): Promise<SettingsResponse> => {
  const { data } = await client.post<SettingsResponse>(`/admin/settings/${chatId}`, payload);
  return data;
};

export const fetchStats = async (chatId: number, window: string): Promise<StatsResponse> => {
  const { data } = await client.get<StatsResponse>(`/admin/stats/${chatId}`, { params: { window } });
  return data;
};

export const logEvent = async (payload: EventPayload) => {
  await client.post("/admin/events", payload);
};

export const runAdminTest = async (text: string, threshold?: number): Promise<PredictionResponse> => {
  const { data } = await client.post<PredictionResponse>("/admin/test", { text, threshold });
  return data;
};

export const fetchGroups = async (): Promise<GroupSummary[]> => {
  const { data } = await client.get<GroupSummary[]>("/admin/groups");
  return data;
};

export const syncGroup = async (chatId: number, payload: { title?: string; group_type?: string }) => {
  await client.post(`/admin/groups/${chatId}/sync`, payload);
};

export const fetchAdmins = async (chatId: number): Promise<AdminEntry[]> => {
  const { data } = await client.get<AdminEntry[]>(`/admin/groups/${chatId}/admins`);
  return data;
};

export const addAdmin = async (chatId: number, userId: number) => {
  await client.post(`/admin/groups/${chatId}/admins`, { user_id: userId });
};

export const removeAdmin = async (chatId: number, userId: number) => {
  await client.delete(`/admin/groups/${chatId}/admins/${userId}`);
};

export const fetchMemberModerations = async (chatId: number, status?: MemberStatus): Promise<MemberModeration[]> => {
  const { data } = await client.get<MemberModeration[]>(`/admin/groups/${chatId}/members`, { params: { status } });
  return data;
};

export const upsertMemberModeration = async (
  chatId: number,
  payload: { user_id: number; username?: string; status: MemberStatus; duration_minutes?: number; reason?: string },
) => {
  await client.post(`/admin/groups/${chatId}/members`, payload);
};

export const removeMemberModeration = async (chatId: number, userId: number, status: MemberStatus) => {
  await client.delete(`/admin/groups/${chatId}/members/${userId}`, { params: { status } });
};

export const fetchActionCount = async (chatId: number, userId: number, action: string, period: "day" | "all" = "day") => {
  const { data } = await client.get<{ count: number }>(`/admin/action_count/${chatId}/${userId}`, { params: { action, period } });
  return data.count;
};
