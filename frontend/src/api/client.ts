import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const AUTH_TOKEN_KEY = 'novel_agent_auth_token';
const API_KEY_STORAGE_KEY = 'novel_agent_api_key';

export type AuthUser = {
  user_id: string;
  username: string;
  display_name: string;
  email?: string;
  api_key?: string;
  created_at?: string;
  updated_at?: string;
  last_login_at?: string;
};

export const getAuthToken = () => localStorage.getItem(AUTH_TOKEN_KEY);
export const setAuthToken = (token: string) => localStorage.setItem(AUTH_TOKEN_KEY, token);
export const clearAuthToken = () => localStorage.removeItem(AUTH_TOKEN_KEY);
export const getApiKey = () => localStorage.getItem(API_KEY_STORAGE_KEY);
export const setApiKey = (apiKey: string) => localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
export const clearApiKey = () => localStorage.removeItem(API_KEY_STORAGE_KEY);
export const clearAuthCredentials = () => {
  clearAuthToken();
  clearApiKey();
};

const extractApiErrorDetail = (data: unknown): string => {
  if (!data) return '';
  if (typeof data === 'string') return data;
  if (typeof data === 'object') {
    const record = data as Record<string, unknown>;
    const direct = record.error ?? record.message ?? record.detail;
    if (typeof direct === 'string') return direct;
  }
  return '';
};

export const getApiErrorMessage = (error: unknown, fallback = '请求失败，请稍后重试。'): string => {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    const responseDetail = extractApiErrorDetail(error.response?.data);
    const rawMessage = responseDetail || error.message || '';
    const combined = `${rawMessage} ${String(error.cause ?? '')}`.toLowerCase();

    if (
      combined.includes('127.0.0.1:27017') ||
      combined.includes('localhost:27017') ||
      combined.includes('topology description') ||
      combined.includes('autoreconnect')
    ) {
      return '后端数据服务暂不可用，请确认本地 MongoDB 已启动，并监听 127.0.0.1:27017。';
    }

    if (
      !error.response ||
      combined.includes('econnrefused') ||
      combined.includes('network error') ||
      (status === 500 && rawMessage.toLowerCase().includes('request failed with status code 500'))
    ) {
      return '后端接口暂不可用，请确认本地后端服务已经启动，并监听 127.0.0.1:5006。';
    }

    if (typeof responseDetail === 'string' && responseDetail.trim()) {
      return responseDetail;
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  if (typeof error === 'string' && error.trim()) {
    return error;
  }

  return fallback;
};

export const getApiErrorStatus = (error: unknown): number | null => {
  if (!axios.isAxiosError(error)) return null;
  return error.response?.status ?? null;
};

export const isApiNotFoundError = (error: unknown, detailIncludes?: string): boolean => {
  if (!axios.isAxiosError(error)) return false;
  if (error.response?.status !== 404) return false;
  if (!detailIncludes) return true;
  const detail = extractApiErrorDetail(error.response?.data);
  return detail.includes(detailIncludes);
};

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  const apiKey = getApiKey();
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

export const api = {
  // System
  getHealth: () => apiClient.get('/api/system/health'),

  // Auth
  register: (data: { username: string; password: string; display_name?: string; email?: string }) =>
    apiClient.post<{ status: string; token: string; api_key: string; user: AuthUser }>('/api/auth/register', data),

  login: (data: { username: string; password: string }) =>
    apiClient.post<{ status: string; token: string; api_key: string; user: AuthUser }>('/api/auth/login', data),

  getCurrentUser: () =>
    apiClient.get<{ status: string; user: AuthUser }>('/api/auth/me'),

  logout: () => apiClient.post('/api/auth/logout'),
  
  // Lore
  getLore: (params?: { worldview_id?: string; outline_id?: string }) => 
    apiClient.get('/api/lore/all', { params }),

  listLore: (params: {
    worldview_id?: string;
    outline_id?: string;
    chapter_outline_id?: string;
    chapter_outline_mode?: 'root' | 'child';
    novel_id?: string;
    world_id?: string;
    type?: string;
    query?: string;
    page: number;
    page_size: number;
  }) =>
    apiClient.get('/api/lore/list', { params }),

  listWorldviews: (params: { world_id?: string; worldview_id?: string; query?: string; page: number; page_size: number }) =>
    apiClient.get('/api/worldviews/list', { params }),

  listWorlds: () => apiClient.get('/api/worlds/list'),

  getWorld: (params: { world_id: string }) =>
    apiClient.get('/api/worlds/get', { params }),

  createWorld: (data: { name: string; summary?: string; forbidden_rules?: string[]; basic_settings?: Record<string, unknown> }) =>
    apiClient.post('/api/worlds/create', data),

  updateWorld: (data: { world_id: string; name?: string; summary?: string; forbidden_rules?: string[]; basic_settings?: Record<string, unknown> }) =>
    apiClient.post('/api/worlds/update', data),

  deleteWorld: (data: { world_id: string; cascade?: boolean }) =>
    apiClient.delete('/api/worlds/delete', { data }),

  updateWorldview: (data: { worldview_id: string; name?: string; summary?: string; world_id?: string }) =>
    apiClient.post('/api/worldviews/update', data),

  listNovels: (params: { world_id?: string; novel_id?: string; query?: string; page: number; page_size: number }) =>
    apiClient.get('/api/novels/list', { params }),

  getNovel: (params: { novel_id: string }) =>
    apiClient.get('/api/novels/get', { params }),

  createNovel: (data: { name: string; introduction?: string; summary?: string; world_id: string; forbidden_rules?: string[]; basic_settings?: Record<string, unknown> }) =>
    apiClient.post('/api/novels/create', data),

  updateNovel: (data: { novel_id: string; name?: string; introduction?: string; summary?: string; world_id?: string; forbidden_rules?: string[]; basic_settings?: Record<string, unknown> }) =>
    apiClient.post('/api/novels/update', data),

  deleteNovel: (data: { novel_id: string; cascade?: boolean }) =>
    apiClient.delete('/api/novels/delete', { data }),

  getWorldHierarchyTree: (params: { world_id: string; worldview_id?: string; novel_id?: string; outline_id?: string; page: number; page_size: number }) =>
    apiClient.get('/api/world-hierarchy/tree', { params }),

  startHierarchyAgent: (data: {
    agent_type:
      | 'world'
      | 'worldview'
      | 'novel'
      | 'outline'
      | 'chapter'
      | 'outline_summary_create'
      | 'outline_summary_update'
      | 'chapter_outline_summary_create'
      | 'chapter_outline_summary_update'
      | 'chapter_content_summary_create'
      | 'chapter_content_summary_update';
    action: 'create' | 'update' | 'check' | 'delete';
    message?: string;
    payload: Record<string, unknown>;
  }) => apiClient.post('/api/hierarchy-agent/start', data),

  respondHierarchyAgent: (data: {
    run_id: string;
    decision: 'approve' | 'request_changes' | 'reject';
    message?: string;
    revision_mode?: 'partial_rewrite' | 'full_rewrite' | 'content_rewrite';
    manual_edit?: boolean;
    payload?: Record<string, unknown>;
  }) => apiClient.post('/api/hierarchy-agent/respond', data),

  getHierarchyAgent: (params: { run_id: string }) =>
    apiClient.get('/api/hierarchy-agent/get', { params }),

  listHierarchyAgents: (params: { agent_type?: string; status?: string; run_id?: string; page: number; page_size: number }) =>
    apiClient.get('/api/hierarchy-agent/list', { params }),

  listDownstreamSummaries: (params: {
    summary_id?: string;
    summary_key?: string;
    agent_type?: string;
    summary_scope?: string;
    summary_action?: string;
    world_id?: string;
    worldview_id?: string;
    novel_id?: string;
    outline_id?: string;
    chapter_id?: string;
    target_id?: string;
    page: number;
    page_size: number;
  }) => apiClient.get('/api/downstream-summaries/list', { params }),

  listOutlines: (params: { world_id?: string; worldview_id?: string; novel_id?: string; outline_id?: string; id?: string; query?: string; page: number; page_size: number }) =>
    apiClient.get('/api/outlines/list', { params }),

  createOutline: (data: { name: string; summary?: string; worldview_id?: string; novel_id?: string; world_id?: string }) =>
    apiClient.post('/api/outlines/create', data),

  updateArchiveItem: (data: {
    id: string;
    type: 'worldview' | 'outline' | 'prose' | 'novel' | 'entity-draft';
    name?: string;
    content?: string;
    category?: string;
    path?: string;
    world_id?: string;
    worldview_id?: string;
    novel_id?: string;
    outline_id?: string;
    chapter_outline_id?: string;
  }) => apiClient.post('/api/archive/update', data),

  deleteArchiveItem: (data: {
    id: string;
    type: 'worldview' | 'outline' | 'prose' | 'novel' | 'entity-draft';
    world_id?: string;
    worldview_id?: string;
    novel_id?: string;
    outline_id?: string;
    chapter_outline_id?: string;
    cascade?: boolean;
  }) => apiClient.delete('/api/archive/delete', { data }),

  searchArchive: (data: { query: string; worldview_id?: string; outline_id?: string }) =>
    apiClient.post('/api/search', data),

  getOutlineChapterState: (params: { world_id?: string; worldview_id?: string; outline_id?: string; page: number; page_size: number }) =>
    apiClient.get('/api/workflow/outline-chapter/state', { params }),
};
