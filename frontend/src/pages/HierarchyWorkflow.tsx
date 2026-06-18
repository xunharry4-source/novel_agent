import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  Node,
  Edge,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Code,
  Divider,
  Grid,
  Group,
  Loader,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Tabs,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconCircleCheck,
  IconCircleX,
  IconEdit,
  IconPlayerPlay,
  IconRefresh,
  IconSend,
} from '@tabler/icons-react';
import { api, getApiErrorMessage } from '../api/client';
import { getWorldviewDisplayName } from '../utils/worldview';

type AgentType =
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
type AgentAction = 'create' | 'update' | 'check';
type RevisionMode = 'partial_rewrite' | 'content_rewrite' | 'full_rewrite';

type WorkflowNode = {
  node_id: string;
  label: string;
  step_index?: number;
  step_title?: string;
  function?: string;
  description?: string;
  status: string;
  input: unknown;
  output: unknown;
  started_at?: string;
  completed_at?: string;
};

type ConversationItem = {
  role: string;
  message?: string;
  payload?: unknown;
  decision?: string;
  revision_mode?: string;
  manual_edit?: boolean;
  review_errors?: string[];
  commit_result?: unknown;
  created_at?: string;
};

type AgentRun = {
  run_id: string;
  agent_type: AgentType;
  agent_name: string;
  action: AgentAction;
  status: string;
  current_node?: string;
  iterations: number;
  review_required: boolean;
  pending_payload: Record<string, unknown>;
  conversation?: ConversationItem[];
  nodes: WorkflowNode[];
  commit_result?: Record<string, unknown>;
  committed?: boolean;
  created_at?: string;
  updated_at?: string;
};

type World = { world_id: string; name: string; summary?: string };
type Worldview = { worldview_id: string; world_id: string; name?: string; title?: string; summary?: string };
type Novel = { novel_id: string; world_id: string; name: string; introduction?: string; summary?: string; forbidden_rules?: unknown; basic_settings?: unknown };
type Outline = { outline_id: string; world_id: string; novel_id?: string; worldview_id?: string; title: string; summary?: string };
type Chapter = { id?: string; scene_id?: string; prose_id?: string; world_id?: string; novel_id?: string; worldview_id?: string; outline_id?: string; name: string; title?: string; content?: string; type?: string };

const chapterRecordId = (chapter: Chapter) => chapter.id || chapter.scene_id || chapter.prose_id || '';

const uniqueWorldsById = (items: World[]): World[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (!item.world_id || seen.has(item.world_id)) return false;
    seen.add(item.world_id);
    return true;
  });
};

const validTypes: AgentType[] = [
  'world',
  'worldview',
  'novel',
  'outline',
  'chapter',
  'outline_summary_create',
  'outline_summary_update',
  'chapter_outline_summary_create',
  'chapter_outline_summary_update',
  'chapter_content_summary_create',
  'chapter_content_summary_update',
];
const validActions: AgentAction[] = ['create', 'update', 'check'];

const typeLabel: Record<AgentType, string> = {
  world: '世界',
  worldview: '世界观设定',
  novel: '小说',
  outline: '大纲',
  chapter: '章节',
  outline_summary_create: '新增分卷大纲总结',
  outline_summary_update: '修改分卷大纲总结',
  chapter_outline_summary_create: '新增章节大纲总结',
  chapter_outline_summary_update: '修改章节大纲总结',
  chapter_content_summary_create: '新增章节内容总结',
  chapter_content_summary_update: '修改章节内容总结',
};

const outlineLikeTypes = new Set<AgentType>(['outline', 'outline_summary_create', 'outline_summary_update']);
const chapterLikeTypes = new Set<AgentType>([
  'chapter',
  'chapter_outline_summary_create',
  'chapter_outline_summary_update',
  'chapter_content_summary_create',
  'chapter_content_summary_update',
]);
const summaryWorkflowTypes = new Set<AgentType>([
  'outline_summary_create',
  'outline_summary_update',
  'chapter_outline_summary_create',
  'chapter_outline_summary_update',
  'chapter_content_summary_create',
  'chapter_content_summary_update',
]);
const chapterSummaryTypes = new Set<AgentType>([
  'chapter_outline_summary_create',
  'chapter_outline_summary_update',
  'chapter_content_summary_create',
  'chapter_content_summary_update',
]);

const actionLabel: Record<AgentAction, string> = {
  create: '新增',
  update: '修改',
  check: '检查',
};

const revisionOptions: Array<{ value: RevisionMode; label: string }> = [
  { value: 'partial_rewrite', label: '局部重写' },
  { value: 'content_rewrite', label: '小部分修改' },
  { value: 'full_rewrite', label: '完全重写' },
];

const workflowStepInfo: Record<string, Pick<WorkflowNode, 'step_index' | 'step_title' | 'function' | 'description'>> = {
  input: {
    step_index: 1,
    step_title: '步骤 1：接收输入',
    function: '接收用户消息与业务 payload',
    description: '记录本次要新增或修改的实体、父级关系和人工意图，作为后续 Agent 草案生成的唯一输入基准。',
  },
  draft: {
    step_index: 2,
    step_title: '步骤 2：LLM 生成草案',
    function: '调用对应业务 Agent 的 LLM',
    description: '根据实体类型调用对应 Agent，扩充 summary/content 并保留关键 ID。',
  },
  revision: {
    step_index: 2,
    step_title: '步骤 2：LLM 迭代草案',
    function: '按人工反馈重新调用 LLM',
    description: '根据修改模式生成下一版草案，并保护未点名字段和父级关系。',
  },
  review: {
    step_index: 3,
    step_title: '步骤 3：业务审查',
    function: '校验草案是否满足层级约束',
    description: '检查必填字段、父级实体、章节归属和跨模块规则；世界模块跳过该步骤。',
  },
  human: {
    step_index: 4,
    step_title: '步骤 4：人工确认',
    function: '等待用户批准、修改或中止',
    description: '草案在这里暂停，不会写入数据库；用户可批准写库或触发下一轮 LLM 迭代。',
  },
  human_response: {
    step_index: 4,
    step_title: '步骤 4：记录人工反馈',
    function: '保存用户决策',
    description: '记录批准、修改或中止决定，以及修改模式、人工编辑内容和反馈文本。',
  },
  apply: {
    step_index: 5,
    step_title: '步骤 5：真实写库',
    function: '执行数据库写入',
    description: '仅在人工批准后创建或更新世界、世界观、小说、大纲或章节，并返回真实写库结果。',
  },
  world_review: {
    step_index: 2,
    step_title: '步骤 2：检查世界规则',
    function: '检查是否违反世界规则',
    description: '读取当前世界设定，判断直接内容是否违反世界级别的硬规则、禁忌与基础约束。',
  },
  novel_review: {
    step_index: 3,
    step_title: '步骤 3：检查小说规则',
    function: '检查是否违反小说规则',
    description: '对照小说级设定、禁则和基调要求，判断内容是否偏离当前小说的写作约束。',
  },
  worldview_review: {
    step_index: 4,
    step_title: '步骤 4：检查世界观',
    function: '检查是否违反世界观设定',
    description: '对照世界观设定库，检查角色、组织、力量体系、历史事实等是否出现设定冲突。',
  },
  outline_review: {
    step_index: 5,
    step_title: '步骤 5：检查分卷大纲',
    function: '检查是否违反分卷大纲',
    description: '核对当前分卷目标、关键情节和推进顺序，检查直接内容是否偏离分卷规划。',
  },
  chapter_outline_review: {
    step_index: 6,
    step_title: '步骤 6：检查章节大纲',
    function: '检查是否违反章节大纲',
    description: '直接对照章节大纲，检查本次内容是否遗漏关键点、顺序错乱或擅自扩写到其他章节。',
  },
  plot_review: {
    step_index: 7,
    step_title: '步骤 7：检查剧情错误',
    function: '检查是否存在剧情错误',
    description: '综合判断是否出现因果错误、人物动机断裂、前后矛盾、时间线错误或关键剧情漏洞。',
  },
  output: {
    step_index: 8,
    step_title: '步骤 8：输出检查结果',
    function: '汇总所有检查结论',
    description: '输出总通过状态、失败检查项、逐项说明与最终检查结论，不执行任何写库操作。',
  },
};

const workflowStepKey = (nodeId: string) => {
  if (nodeId.startsWith('draft_retry')) return 'draft';
  if (nodeId.startsWith('revision_retry')) return 'revision';
  if (nodeId.startsWith('review_retry')) return 'review';
  return nodeId;
};

const withStepInfo = (node: WorkflowNode): WorkflowNode => ({
  ...(workflowStepInfo[workflowStepKey(node.node_id)] || {}),
  ...node,
});

const plannedNodes: WorkflowNode[] = [
  { node_id: 'input', label: '输入节点', status: 'pending', input: {}, output: {} },
  { node_id: 'draft', label: '草案节点', status: 'pending', input: {}, output: {} },
  { node_id: 'review', label: '审查节点', status: 'pending', input: {}, output: {} },
  { node_id: 'human', label: '人工介入', status: 'pending', input: {}, output: {} },
  { node_id: 'apply', label: '写库节点', status: 'pending', input: {}, output: {} },
].map(withStepInfo);

const worldPlannedNodes: WorkflowNode[] = [
  { node_id: 'input', label: '输入节点', status: 'pending', input: {}, output: {} },
  { node_id: 'draft', label: '创世草案节点', status: 'pending', input: {}, output: {} },
  { node_id: 'human', label: '人工介入', status: 'pending', input: {}, output: {} },
  { node_id: 'apply', label: '写库固化节点', status: 'pending', input: {}, output: {} },
].map(withStepInfo);

const chapterCheckPlannedNodes: WorkflowNode[] = [
  { node_id: 'input', label: '输入节点', status: 'pending', input: {}, output: {} },
  { node_id: 'world_review', label: '世界规则检查', status: 'pending', input: {}, output: {} },
  { node_id: 'novel_review', label: '小说规则检查', status: 'pending', input: {}, output: {} },
  { node_id: 'worldview_review', label: '世界观检查', status: 'pending', input: {}, output: {} },
  { node_id: 'outline_review', label: '分卷大纲检查', status: 'pending', input: {}, output: {} },
  { node_id: 'chapter_outline_review', label: '章节大纲检查', status: 'pending', input: {}, output: {} },
  { node_id: 'plot_review', label: '剧情错误检查', status: 'pending', input: {}, output: {} },
  { node_id: 'output', label: '检查结果输出', status: 'pending', input: {}, output: {} },
].map(withStepInfo);

const statusColor: Record<string, string> = {
  pending: '#495057',
  waiting: '#f08c00',
  waiting_human: '#f08c00',
  blocked: '#e03131',
  running: '#228be6',
  completed: '#2f9e44',
  skipped: '#868e96',
  failed: '#e03131',
  review_failed: '#e03131',
  rejected: '#868e96',
};

const stringify = (value: unknown) => JSON.stringify(value ?? {}, null, 2);
const parseJsonParam = (value: string | null): unknown => {
  if (!value) return undefined;
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
};
const asRecord = (value: unknown): Record<string, any> => (
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, any> : {}
);

const first = (value: string | null, fallback = '') => value || fallback;

const deriveOutlineNameFromSummary = (rawSummary: string): string => {
  const lines = rawSummary
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return '';
  const volumeLine = lines.find((line) => /^第.{0,12}卷(?:[：:].*)?$/.test(line) || /^第.{0,12}卷[：:]\s*.+$/.test(line));
  if (volumeLine) return volumeLine;
  const shortLine = lines.find((line) => line.length <= 40);
  return shortLine || lines[0];
};

type HierarchyWorkflowProps = {
  fixedType?: AgentType;
};

export const HierarchyWorkflow: React.FC<HierarchyWorkflowProps> = ({ fixedType }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);

  const requestedType = (fixedType || params.get('type')) as AgentType | null;
  const requestedAction = params.get('action') as AgentAction | null;
  const agentType = validTypes.includes(requestedType as AgentType) ? requestedType as AgentType : 'world';
  const action = validActions.includes(requestedAction as AgentAction) ? requestedAction as AgentAction : 'create';
  const targetId = params.get('id') || params.get('target_id') || '';
  const runId = params.get('run_id') || '';
  const isWorldWorkflow = agentType === 'world';
  const isOutlineLike = outlineLikeTypes.has(agentType);
  const isChapterLike = chapterLikeTypes.has(agentType);
  const isChapterCheck = agentType === 'chapter' && action === 'check';
  const isSummaryWorkflow = summaryWorkflowTypes.has(agentType);

  const [worlds, setWorlds] = useState<World[]>([]);
  const [worldviews, setWorldviews] = useState<Worldview[]>([]);
  const [novels, setNovels] = useState<Novel[]>([]);
  const [outlines, setOutlines] = useState<Outline[]>([]);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [selectedWorldId, setSelectedWorldId] = useState(isWorldWorkflow ? '' : first(params.get('world_id'), ''));
  const [selectedWorldviewId, setSelectedWorldviewId] = useState(first(params.get('worldview_id'), ''));
  const [selectedNovelId, setSelectedNovelId] = useState(first(params.get('novel_id'), ''));
  const [selectedOutlineId, setSelectedOutlineId] = useState(first(params.get('outline_id'), ''));
  const [parentPath, setParentPath] = useState(first(params.get('parent_path'), ''));
  const [currentPath, setCurrentPath] = useState(first(params.get('path'), ''));
  const [name, setName] = useState(first(params.get('name'), ''));
  const [introduction, setIntroduction] = useState(first(params.get('introduction'), ''));
  const [summary, setSummary] = useState(first(params.get('summary'), ''));
  const [content, setContent] = useState(first(params.get('content'), ''));
  const [selectedChapterId, setSelectedChapterId] = useState(first(params.get('chapter_outline_id') || params.get('target_id') || params.get('id'), ''));
  const [chapterOutline, setChapterOutline] = useState(first(params.get('chapter_outline'), ''));
  const [message, setMessage] = useState(first(params.get('message'), `${actionLabel[action]}${typeLabel[agentType]}`));
  const [feedback, setFeedback] = useState('');
  const [revisionMode, setRevisionMode] = useState<RevisionMode>('partial_rewrite');
  const [run, setRun] = useState<AgentRun | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState('input');
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const outlineNameSuggestion = useMemo(
    () => (isOutlineLike ? deriveOutlineNameFromSummary(summary) : ''),
    [isOutlineLike, summary],
  );
  const effectiveName = useMemo(
    () => name.trim() || (isOutlineLike ? outlineNameSuggestion : ''),
    [isOutlineLike, name, outlineNameSuggestion],
  );

  useEffect(() => {
    if (!isWorldWorkflow || !params.get('world_id')) return;
    const next = new URLSearchParams(location.search);
    next.delete('world_id');
    navigate(`${location.pathname}?${next.toString()}`, { replace: true });
  }, [isWorldWorkflow, location.pathname, location.search, navigate, params]);

  const applyRunPayload = useCallback((nextRun: AgentRun) => {
    const payload = nextRun.pending_payload || {};
    if (typeof payload.name === 'string') {
      setName(payload.name);
    }
    if (typeof payload.introduction === 'string') {
      setIntroduction(payload.introduction);
    }
    if (isChapterLike) {
      if (typeof payload.content === 'string') {
        setContent(payload.content);
      }
      if (typeof payload.chapter_outline === 'string') {
        setChapterOutline(payload.chapter_outline);
      }
      if (typeof payload.chapter_outline_id === 'string') {
        setSelectedChapterId(payload.chapter_outline_id);
      } else if (typeof payload.target_id === 'string' && isChapterCheck) {
        setSelectedChapterId(payload.target_id);
      }
    } else if (typeof payload.summary === 'string') {
      setSummary(payload.summary);
    }
    if (typeof payload.world_id === 'string' && !isWorldWorkflow) {
      setSelectedWorldId(payload.world_id);
    }
    if (typeof payload.worldview_id === 'string') {
      setSelectedWorldviewId(payload.worldview_id);
    }
    if (typeof payload.parent_path === 'string') {
      setParentPath(payload.parent_path);
    }
    if (typeof payload.path === 'string') {
      setCurrentPath(payload.path);
    }
    if (typeof payload.novel_id === 'string') {
      setSelectedNovelId(payload.novel_id);
    }
    if (typeof payload.outline_id === 'string') {
      setSelectedOutlineId(payload.outline_id);
    }
  }, [isChapterCheck, isChapterLike, isWorldWorkflow]);

  const resolveRunNodeId = useCallback((nextRun: AgentRun) => {
    const nodes = isWorldWorkflow ? nextRun.nodes.filter((node) => node.node_id !== 'review') : nextRun.nodes;
    const current = isWorldWorkflow && nextRun.current_node === 'review' ? 'human' : nextRun.current_node;
    if (current && nodes.some((node) => node.node_id === current)) {
      return current;
    }
    return nodes[nodes.length - 1]?.node_id || 'input';
  }, [isWorldWorkflow]);

  const loadRun = useCallback(async (id: string) => {
    if (!id) return;
    const response = await api.getHierarchyAgent({ run_id: id });
    const nextRun = response.data.run as AgentRun;
    setRun(nextRun);
    applyRunPayload(nextRun);
    setSelectedNodeId(resolveRunNodeId(nextRun));
  }, [applyRunPayload, resolveRunNodeId]);

  const loadContext = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (isWorldWorkflow && action === 'create' && !runId) {
        return;
      }

      const worldRes = await api.listWorlds();
      const nextWorlds = uniqueWorldsById((worldRes.data as World[]) || []);
      setWorlds(nextWorlds);
      const nextWorldId = isWorldWorkflow ? '' : selectedWorldId || params.get('world_id') || nextWorlds[0]?.world_id || '';
      setSelectedWorldId(nextWorldId);

      const [wvRes, novelRes, outlineRes] = !isWorldWorkflow && nextWorldId
        ? await Promise.all([
          api.listWorldviews({ world_id: nextWorldId, page: 1, page_size: 100 }),
          api.listNovels({ world_id: nextWorldId, page: 1, page_size: 100 }),
          api.listOutlines({ world_id: nextWorldId, page: 1, page_size: 100 }),
        ])
        : [{ data: [] }, { data: [] }, { data: [] }];

      const nextWorldviews = wvRes.data as Worldview[];
      const nextNovels = novelRes.data as Novel[];
      const nextOutlines = outlineRes.data as Outline[];
      setWorldviews(nextWorldviews);
      setNovels(nextNovels);
      setOutlines(nextOutlines);
      setSelectedWorldviewId((current) => current || params.get('worldview_id') || nextWorldviews[0]?.worldview_id || '');
      setSelectedNovelId((current) => {
        if (current) return current;
        const initialNovelId = params.get('novel_id');
        if (initialNovelId) return initialNovelId;
        return isOutlineLike || isChapterLike ? '' : nextNovels[0]?.novel_id || '';
      });
      setSelectedOutlineId((current) => {
        if (current) return current;
        const initialOutlineId = params.get('outline_id');
        if (initialOutlineId) return initialOutlineId;
        return isChapterLike ? '' : nextOutlines[0]?.outline_id || '';
      });

      const chapterScope = params.get('outline_id') || selectedOutlineId || '';
      if (nextWorldId && chapterScope) {
        const chapterRes = await api.listLore({ world_id: nextWorldId, outline_id: chapterScope, type: 'prose', page: 1, page_size: 100 });
        setChapters((chapterRes.data as Chapter[]).filter((item) => item.type === 'prose'));
      } else {
        setChapters([]);
      }

      if ((action === 'update' || isChapterCheck) && targetId) {
        if (agentType === 'world') {
          const found = nextWorlds.find((item) => item.world_id === targetId);
          if (found) {
            setName(params.get('name') || found.name || '');
            setSummary(params.get('summary') || found.summary || '');
            setSelectedWorldId(found.world_id);
          }
        }
        if (agentType === 'worldview') {
          const res = await api.listWorldviews({ worldview_id: targetId, page: 1, page_size: 10 });
          const found = (res.data as Worldview[]).find((item) => item.worldview_id === targetId);
          if (found) {
            setName(found.name || '');
            setSummary(found.summary || '');
            setSelectedWorldId(found.world_id || nextWorldId);
            setSelectedWorldviewId(found.worldview_id);
          }
        }
        if (agentType === 'novel') {
          const res = await api.listNovels({ novel_id: targetId, page: 1, page_size: 10 });
          const found = (res.data as Novel[]).find((item) => item.novel_id === targetId);
          if (found) {
            setName(params.get('name') || found.name || '');
            setIntroduction(params.get('introduction') || found.introduction || '');
            setSummary(params.get('summary') || found.summary || '');
            setSelectedWorldId(found.world_id || nextWorldId);
            setSelectedNovelId(found.novel_id);
          }
        }
        if (isOutlineLike) {
          const res = await api.listOutlines({ outline_id: targetId, page: 1, page_size: 10 });
          const found = (res.data as Outline[]).find((item) => item.outline_id === targetId);
          if (found) {
            setName(found.title || '');
            setSummary(found.summary || '');
            setSelectedWorldId(found.world_id || nextWorldId);
            setSelectedWorldviewId(found.worldview_id || '');
            setSelectedNovelId(found.novel_id || '');
          }
        }
        if (isChapterLike) {
          const scopeWorld = nextWorldId || selectedWorldId;
          const scopeOutline = params.get('outline_id') || selectedOutlineId;
          if (scopeWorld && scopeOutline) {
            const res = await api.listLore({ world_id: scopeWorld, outline_id: scopeOutline, type: 'prose', page: 1, page_size: 100 });
            const found = (res.data as Chapter[]).find((item) => chapterRecordId(item) === targetId);
            if (found) {
              setName(found.name || found.title || '');
              if (isChapterCheck) {
                setSelectedChapterId(chapterRecordId(found));
                setChapterOutline(found.content || '');
                setContent(params.get('content') || '');
              } else {
                setContent(found.content || '');
              }
              setSelectedWorldId(found.world_id || scopeWorld);
              setSelectedWorldviewId(found.worldview_id || '');
              setSelectedNovelId(found.novel_id || '');
              setSelectedOutlineId(found.outline_id || scopeOutline);
              setChapters((res.data as Chapter[]).filter((item) => item.type === 'prose'));
            }
          }
        }
      }

      if (runId) {
        await loadRun(runId);
      }
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '加载工作流上下文失败，请稍后重试。'));
    } finally {
      setLoading(false);
    }
  }, [action, agentType, isChapterCheck, isChapterLike, isOutlineLike, isWorldWorkflow, loadRun, params, runId, selectedOutlineId, selectedWorldId, targetId]);

  useEffect(() => {
    loadContext();
  }, [loadContext]);

  const handleChapterOutlineSelect = useCallback((value: string | null) => {
    const nextId = value || '';
    setSelectedChapterId(nextId);
    if (!nextId) return;
    const targetChapter = chapters.find((item) => chapterRecordId(item) === nextId);
    if (!targetChapter) return;
    setName(targetChapter.name || targetChapter.title || '');
    setChapterOutline(targetChapter.content || '');
    if (targetChapter.world_id) setSelectedWorldId(targetChapter.world_id);
    if (targetChapter.worldview_id) setSelectedWorldviewId(targetChapter.worldview_id);
    if (targetChapter.novel_id) setSelectedNovelId(targetChapter.novel_id);
    if (targetChapter.outline_id) setSelectedOutlineId(targetChapter.outline_id);
  }, [chapters]);

  const selectedNodes = useMemo(() => {
    const rawNodes = run?.nodes?.length
      ? run.nodes.map(withStepInfo)
      : (isWorldWorkflow ? worldPlannedNodes : isChapterCheck ? chapterCheckPlannedNodes : plannedNodes);
    return isWorldWorkflow ? rawNodes.filter((node) => node.node_id !== 'review') : rawNodes;
  }, [isChapterCheck, isWorldWorkflow, run?.nodes]);
  const rawCurrentNodeId = run?.current_node || selectedNodes.find((node) => ['waiting', 'blocked', 'failed'].includes(node.status))?.node_id || selectedNodeId;
  const currentNodeId = isWorldWorkflow && rawCurrentNodeId === 'review' ? 'human' : rawCurrentNodeId;
  const selectedNode = selectedNodes.find((node) => node.node_id === selectedNodeId) || selectedNodes[0];
  const selectedNodeOutput = asRecord(selectedNode?.output);
  const selectedLlmCall = asRecord(selectedNodeOutput.llm_call);
  const selectedLlmPrompt = typeof selectedLlmCall.prompt === 'string' ? selectedLlmCall.prompt : '';

  const graph = useMemo(() => {
    const nodes: Node[] = selectedNodes.map((node, index) => {
      const color = statusColor[node.status] || '#228be6';
      return {
        id: node.node_id,
        position: { x: 60 + index * 250, y: 120 },
        data: { label: `${node.step_title || node.label}\n${node.function || node.label}\n${node.status}` },
        style: {
          width: 200,
          minHeight: 88,
          whiteSpace: 'pre-line',
          borderRadius: 8,
          border: node.node_id === currentNodeId ? `2px solid ${color}` : `1px solid ${color}`,
          background: node.node_id === currentNodeId ? '#102a24' : '#141517',
          color: '#f8f9fa',
          fontSize: 12,
          fontWeight: 700,
          boxShadow: node.node_id === currentNodeId ? `0 0 18px ${color}66` : 'none',
        },
      };
    });
    const edges: Edge[] = nodes.slice(1).map((node, index) => ({
      id: `edge-${index}`,
      source: nodes[index].id,
      target: node.id,
      animated: nodes[index].id === currentNodeId,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { stroke: '#4dabf7' },
    }));
    return { nodes, edges };
  }, [currentNodeId, selectedNodes]);

  const buildPayload = (): Record<string, unknown> => {
    const payload: Record<string, unknown> = {};
    if (action === 'update') {
      payload.target_id = targetId;
    }
    if (isSummaryWorkflow && targetId) {
      payload.target_id = targetId;
      if (chapterSummaryTypes.has(agentType)) {
        payload.chapter_id = targetId;
      }
    }
    if (!isChapterLike) {
      payload.name = effectiveName;
      if (agentType === 'novel' && introduction) {
        payload.introduction = introduction;
      }
      payload.summary = summary;
    } else {
      if (name.trim()) {
        payload.name = name.trim();
      }
      payload.content = content || summary;
    }
    if (agentType === 'worldview' || agentType === 'novel') {
      payload.world_id = selectedWorldId;
    }
    if (agentType === 'worldview' && selectedWorldviewId) {
      payload.worldview_id = selectedWorldviewId;
      if (action === 'create' && parentPath.trim()) {
        payload.parent_path = parentPath.trim();
      }
    }
    if (agentType === 'world' || agentType === 'novel') {
      const forbiddenRules = parseJsonParam(params.get('forbidden_rules'));
      const basicSettings = parseJsonParam(params.get('basic_settings'));
      if (forbiddenRules !== undefined) payload.forbidden_rules = forbiddenRules;
      if (basicSettings !== undefined) payload.basic_settings = basicSettings;
    }
    if (isOutlineLike) {
      payload.novel_id = selectedNovelId;
      payload.worldview_id = selectedWorldviewId;
      payload.world_id = selectedWorldId;
    }
    if (isChapterLike) {
      payload.outline_id = selectedOutlineId;
      payload.novel_id = selectedNovelId;
      payload.worldview_id = selectedWorldviewId;
      payload.world_id = selectedWorldId;
      if (selectedChapterId) {
        payload.chapter_outline_id = selectedChapterId;
      }
      if (isChapterCheck) {
        payload.chapter_outline = chapterOutline;
        if (selectedChapterId) {
          payload.target_id = selectedChapterId;
          payload.chapter_outline_id = selectedChapterId;
        }
      }
    }
    return payload;
  };

  const requiredContextError = useMemo(() => {
    if (agentType === 'world') return '';
    if (!selectedWorldId) return `${typeLabel[agentType]}工作流必须提供 world_id`;
    if (agentType === 'worldview' && !selectedWorldviewId) return '世界观工作流必须绑定当前世界唯一的世界观设定库';
    if (isOutlineLike && !selectedNovelId) return `${typeLabel[agentType]}工作流必须提供 novel_id`;
    if (isChapterLike && !selectedOutlineId) return `${typeLabel[agentType]}工作流必须提供 outline_id`;
    return '';
  }, [agentType, isChapterLike, isOutlineLike, selectedNovelId, selectedOutlineId, selectedWorldId, selectedWorldviewId]);

  const validateBeforeStart = useCallback(() => {
    if (!fixedType && !validTypes.includes(requestedType as AgentType)) return `非法 type: ${requestedType}`;
    if (!validActions.includes(requestedAction as AgentAction)) return `非法 action: ${requestedAction}`;
    if (action === 'update' && !targetId) return '修改工作流必须提供 id 或 target_id';
    if (requiredContextError) return requiredContextError;
    if (!isChapterCheck && !effectiveName) {
      if (isOutlineLike) {
        return '分卷大纲必须提供卷名；你可以直接在正文首行写“第一卷：失踪者”，系统会自动识别。';
      }
      return `${typeLabel[agentType]}名称不能为空`;
    }
    if ((agentType === 'worldview' || agentType === 'novel') && !selectedWorldId) return `${typeLabel[agentType]}必须选择世界`;
    if (isOutlineLike && !selectedNovelId) return `${typeLabel[agentType]}必须选择小说`;
    if (isChapterLike && !selectedOutlineId) return `${typeLabel[agentType]}必须选择大纲`;
    if (isChapterCheck && !chapterOutline.trim()) return '章节检查工作流必须提供章节大纲';
    if (isChapterLike && !(content || summary).trim()) return `${typeLabel[agentType]}内容不能为空`;
    return '';
  }, [
    action,
    agentType,
    chapterOutline,
    content,
    effectiveName,
    fixedType,
    isChapterCheck,
    isChapterLike,
    isOutlineLike,
    requiredContextError,
    requestedAction,
    requestedType,
    selectedChapterId,
    selectedNovelId,
    selectedOutlineId,
    selectedWorldId,
    targetId,
  ]);
  const startValidationError = useMemo(() => (run ? '' : validateBeforeStart()), [run, validateBeforeStart]);

  const replaceRunId = (nextRunId: string) => {
    const next = new URLSearchParams(location.search);
    next.set('run_id', nextRunId);
    navigate(`${location.pathname}?${next.toString()}`, { replace: true });
  };

  const startWorkflow = async () => {
    const validationError = validateBeforeStart();
    if (validationError) {
      setError(validationError);
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await api.startHierarchyAgent({
        agent_type: agentType,
        action,
        message,
        payload: buildPayload(),
      });
      const nextRun = response.data.run as AgentRun;
      setRun(nextRun);
      applyRunPayload(nextRun);
      setSelectedNodeId(resolveRunNodeId(nextRun));
      replaceRunId(nextRun.run_id);
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '启动工作流失败，请稍后重试。'));
    } finally {
      setSubmitting(false);
    }
  };

  const respondWorkflow = async (decision: 'approve' | 'request_changes' | 'reject') => {
    if (!run) return;
    setSubmitting(true);
    setError(null);
    try {
      const response = await api.respondHierarchyAgent({
        run_id: run.run_id,
        decision,
        message: feedback,
        revision_mode: decision === 'request_changes' ? revisionMode : undefined,
        manual_edit: decision === 'request_changes',
        payload: decision === 'request_changes' ? buildPayload() : undefined,
      });
      const nextRun = response.data.run as AgentRun;
      setRun(nextRun);
      applyRunPayload(nextRun);
      setSelectedNodeId(resolveRunNodeId(nextRun));
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '提交人工决策失败，请稍后重试。'));
    } finally {
      setSubmitting(false);
    }
  };

  const title = `${actionLabel[action]}${typeLabel[agentType]}工作流`;
  const isWorldCreate = isWorldWorkflow && action === 'create';
  const contentRows = isWorldCreate ? 9 : isChapterLike ? 8 : 6;
  const messageRows = isWorldCreate ? 5 : 3;
  const feedbackRows = isWorldCreate ? 6 : 4;
  const worldviewPlacementText = useMemo(() => {
    if (agentType !== 'worldview') return '';
    if (action === 'update') {
      return currentPath.trim() || '当前条目未记录层级路径';
    }
    const nextName = name.trim() || '新设定名称';
    return parentPath.trim() ? `${parentPath.trim()} > ${nextName}` : nextName;
  }, [action, agentType, currentPath, name, parentPath]);

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-start">
        <Group>
          <ActionIcon variant="light" onClick={() => navigate(-1)} aria-label="返回">
            <IconArrowLeft size={18} />
          </ActionIcon>
          <Box>
            <Title order={2}>{title}</Title>
            <Text size="sm" c="dimmed">{typeLabel[agentType]}专属 Dify-like 节点流与对话框；所有节点输入输出来自真实 hierarchy_agent_runs。</Text>
          </Box>
        </Group>
        <Group gap="xs">
          <Badge color="blue">{agentType}</Badge>
          <Badge color="cyan">{action}</Badge>
          {run && <Badge color={run.status === 'completed' ? 'green' : run.status === 'review_failed' ? 'red' : 'yellow'}>{run.status}</Badge>}
          {loading && <Loader size="sm" />}
          <Button leftSection={<IconRefresh size={16} />} variant="light" onClick={loadContext}>刷新</Button>
        </Group>
      </Group>

      {error && (
        <Alert color="red" icon={<IconCircleX size={18} />} title="请求失败">
          {error}
        </Alert>
      )}
      {requiredContextError && !run && (
        <Alert color="red" icon={<IconCircleX size={18} />} title="缺少必需上下文">
          {requiredContextError}
        </Alert>
      )}

      <Paper p="md" withBorder>
        <Stack gap="sm">
          <Group justify="space-between">
            <Title order={4}>{actionLabel[action]}表单</Title>
            {isWorldWorkflow && <Badge color="cyan">顶层实体</Badge>}
          </Group>
          <Stack gap="md">
            <Grid gutter="md">
              {!isWorldWorkflow && (
                <Grid.Col span={{ base: 12, md: 6, xl: 3 }}>
                  <Select
                    label="世界"
                    data={worlds.map((world) => ({ value: world.world_id, label: `${world.name} (${world.world_id})` }))}
                    value={selectedWorldId || null}
                    onChange={(value) => setSelectedWorldId(value || '')}
                    disabled={Boolean(run)}
                    searchable
                  />
                </Grid.Col>
              )}
              {(isOutlineLike || isChapterLike) && (
                <Grid.Col span={{ base: 12, md: 6, xl: 3 }}>
                  <Select
                    label="世界观"
                    data={worldviews.map((worldview) => ({ value: worldview.worldview_id, label: getWorldviewDisplayName(worldview) }))}
                    value={selectedWorldviewId || null}
                    onChange={(value) => setSelectedWorldviewId(value || '')}
                    disabled={Boolean(run)}
                    searchable
                  />
                </Grid.Col>
              )}
              {isOutlineLike && (
                <Grid.Col span={{ base: 12, md: 6, xl: 3 }}>
                  <Select
                    label="小说"
                    data={novels.map((novel) => ({ value: novel.novel_id, label: `${novel.name} (${novel.novel_id})` }))}
                    value={selectedNovelId || null}
                    onChange={(value) => setSelectedNovelId(value || '')}
                    disabled={Boolean(run)}
                    searchable
                  />
                </Grid.Col>
              )}
              {isChapterLike && (
                <Grid.Col span={{ base: 12, md: 6, xl: 3 }}>
                  <Select
                    label="大纲"
                    data={outlines.map((outline) => ({ value: outline.outline_id, label: `${outline.title} (${outline.outline_id})` }))}
                    value={selectedOutlineId || null}
                    onChange={(value) => setSelectedOutlineId(value || '')}
                    disabled={Boolean(run)}
                    searchable
                  />
                </Grid.Col>
              )}
              {isChapterCheck && (
                <Grid.Col span={{ base: 12, md: 6, xl: 3 }}>
                  <Select
                    label="章节大纲来源"
                    data={chapters.map((chapter) => ({
                      value: chapterRecordId(chapter),
                      label: `${chapter.name || chapter.title || chapterRecordId(chapter)} (${chapterRecordId(chapter)})`,
                    }))}
                    value={selectedChapterId || null}
                    onChange={handleChapterOutlineSelect}
                    disabled={Boolean(run)}
                    searchable
                    clearable
                    placeholder="可选：从当前分卷章节大纲中带入"
                  />
                </Grid.Col>
              )}
              <Grid.Col span={{ base: 12, md: 6, xl: 3 }}>
                <TextInput
                  label={isChapterCheck ? '检查任务名称（可选）' : isOutlineLike ? '卷名' : '名称'}
                  placeholder={isOutlineLike ? '例如：第一卷：失踪者' : undefined}
                  description={
                    isOutlineLike && !name.trim() && outlineNameSuggestion
                      ? `将自动使用正文中识别到的卷名：${outlineNameSuggestion}`
                      : undefined
                  }
                  value={name}
                  onChange={(event) => setName(event.currentTarget.value)}
                  disabled={Boolean(run?.committed)}
                />
              </Grid.Col>
              {agentType === 'worldview' && (
                <Grid.Col span={{ base: 12, md: 12, xl: 6 }}>
                  <TextInput
                    label={action === 'create' ? '新增后所在层级' : '当前层级路径'}
                    value={worldviewPlacementText}
                    description={
                      action === 'create'
                        ? (parentPath.trim()
                          ? '该层级来自 /worldviews 中你点击的节点或设定条目。'
                          : '未指定父级时会作为顶层世界观设定创建。')
                        : '当前条目的层级路径只读展示；修改内容不会自动移动到其他目录。'
                    }
                    readOnly
                    disabled
                  />
                </Grid.Col>
              )}
            </Grid>

            {agentType === 'novel' && (
              <Textarea label="介绍" minRows={3} autosize value={introduction} onChange={(event) => setIntroduction(event.currentTarget.value)} disabled={Boolean(run?.committed)} />
            )}

            {isChapterCheck ? (
              <Stack gap="md">
                <Textarea
                  label="章节大纲"
                  description="用于对照检查直接内容是否偏离当前章节规划。"
                  minRows={6}
                  autosize
                  value={chapterOutline}
                  onChange={(event) => setChapterOutline(event.currentTarget.value)}
                  disabled={Boolean(run)}
                />
                <Textarea
                  label="直接内容"
                  description="输入需要检查的正文内容；该流程只输出检查结果，不写库。"
                  minRows={contentRows}
                  autosize
                  value={content}
                  onChange={(event) => setContent(event.currentTarget.value)}
                  disabled={Boolean(run)}
                />
              </Stack>
            ) : isChapterLike ? (
              <Textarea label={chapterSummaryTypes.has(agentType) ? '章节原文' : '章节内容'} minRows={contentRows} autosize value={content} onChange={(event) => setContent(event.currentTarget.value)} disabled={Boolean(run?.committed)} />
            ) : (
              <Textarea
                label={isOutlineLike ? (isSummaryWorkflow ? '分卷大纲原文' : '分卷大纲正文') : '摘要/设定'}
                minRows={contentRows}
                autosize
                value={summary}
                onChange={(event) => setSummary(event.currentTarget.value)}
                disabled={Boolean(run?.committed)}
              />
            )}

            <Textarea label="给 Agent 的消息" minRows={messageRows} autosize value={message} onChange={(event) => setMessage(event.currentTarget.value)} disabled={Boolean(run)} />
          </Stack>
          {!run && startValidationError && !requiredContextError && (
            <Alert color="yellow" icon={<IconCircleX size={18} />} title="启动前检查">
              {startValidationError}
            </Alert>
          )}
          <Group justify="flex-end">
            <Button leftSection={<IconPlayerPlay size={16} />} loading={submitting} onClick={startWorkflow} disabled={Boolean(run) || Boolean(requiredContextError)}>
              {isChapterCheck ? '启动检查工作流' : '启动工作流'}
            </Button>
          </Group>
        </Stack>
      </Paper>

      <Paper p="md" withBorder>
        <Stack gap="md">
          <Group justify="space-between">
            <Title order={4}>工作流与节点信息</Title>
            <Group gap="xs">
              <Badge>当前节点：{currentNodeId}</Badge>
              {isChapterCheck
                ? <Badge color={run?.status === 'completed' ? 'green' : 'gray'}>{run?.status === 'completed' ? '已输出检查结果' : '待输出检查结果'}</Badge>
                : run?.committed ? <Badge color="green">已写库</Badge> : <Badge color="gray">未写库</Badge>}
            </Group>
          </Group>
          <Grid gutter="md" align="stretch">
            <Grid.Col span={{ base: 12, lg: 7 }}>
              <Paper h={460} withBorder style={{ overflow: 'hidden' }}>
                <ReactFlow
                  nodes={graph.nodes}
                  edges={graph.edges}
                  onNodeClick={(_, node) => setSelectedNodeId(node.id)}
                  preventScrolling={!isWorldCreate}
                  zoomOnScroll={!isWorldCreate}
                  fitView
                >
                  <Background />
                  <Controls />
                </ReactFlow>
              </Paper>
              <Grid mt="md">
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Paper p="md" withBorder>
                    <Group justify="space-between">
                      <Text fw={800}>当前节点</Text>
                      <Badge>{currentNodeId}</Badge>
                    </Group>
                    <Text size="sm" c="dimmed">迭代轮次：{run?.iterations || 0}</Text>
                    <Text size="sm" c="dimmed">run_id：{run?.run_id || '尚未启动'}</Text>
                  </Paper>
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Paper p="md" withBorder>
                    <Group justify="space-between">
                      <Text fw={800}>{isChapterCheck ? '检查结果' : '写库结果'}</Text>
                      {isChapterCheck
                        ? <Badge color={run?.status === 'completed' ? 'green' : 'gray'}>{run?.status === 'completed' ? '已输出' : '待输出'}</Badge>
                        : run?.committed ? <Badge color="green">已写库</Badge> : <Badge color="gray">未写库</Badge>}
                    </Group>
                    <Text size="sm" c="dimmed">{isChapterCheck ? '该流程只输出检查结论，不执行任何写库操作。' : '人工批准前不会写库。'}</Text>
                  </Paper>
                </Grid.Col>
              </Grid>
            </Grid.Col>

            <Grid.Col span={{ base: 12, lg: 5 }}>
              <Paper p="md" withBorder h="100%" mih={560} style={{ overflow: 'hidden' }}>
            <Tabs defaultValue="steps">
              <Tabs.List grow>
                <Tabs.Tab value="steps">步骤</Tabs.Tab>
                <Tabs.Tab value="chat">对话</Tabs.Tab>
                <Tabs.Tab value="node">节点</Tabs.Tab>
                <Tabs.Tab value="run">Run</Tabs.Tab>
              </Tabs.List>

              <Tabs.Panel value="steps" pt="md">
                <ScrollArea h={420}>
                  <Stack gap="sm">
                    {selectedNodes.map((node, index) => (
                      <Paper
                        key={`${node.node_id}-${index}`}
                        p="sm"
                        withBorder
                        bg={node.node_id === currentNodeId ? 'dark.6' : undefined}
                      >
                        <Group justify="space-between" align="flex-start">
                          <Box>
                            <Text fw={800}>{node.step_title || node.label}</Text>
                            <Text size="sm">{node.function || node.label}</Text>
                          </Box>
                          <Badge color={node.status === 'failed' || node.status === 'blocked' ? 'red' : node.status === 'completed' ? 'green' : 'yellow'}>
                            {node.status}
                          </Badge>
                        </Group>
                        <Text size="sm" c="dimmed" mt="xs">{node.description || '等待节点执行后展示说明。'}</Text>
                      </Paper>
                    ))}
                  </Stack>
                </ScrollArea>
              </Tabs.Panel>

              <Tabs.Panel value="chat" pt="md">
                <ScrollArea h={420}>
                  <Stack gap="sm">
                    {(run?.conversation || []).map((item, index) => (
                      <Alert key={`${item.created_at}-${index}`} color={item.role === 'user' ? 'blue' : 'green'} title={item.role}>
                        <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{item.message || ''}</Text>
                        {Boolean(item.review_errors?.length || item.payload || item.commit_result) && (
                          <Code block mt="xs">{stringify(item.review_errors || item.payload || item.commit_result)}</Code>
                        )}
                      </Alert>
                    ))}
                    {!run && <Text c="dimmed">启动工作流后显示真实对话记录。</Text>}
                  </Stack>
                </ScrollArea>
              </Tabs.Panel>

              <Tabs.Panel value="node" pt="md">
                <ScrollArea h={420}>
                  <Stack gap="sm">
                    <Group justify="space-between">
                      <Text fw={800}>{selectedNode?.label}</Text>
                      <Badge color={selectedNode?.status === 'failed' ? 'red' : selectedNode?.status === 'completed' ? 'green' : 'yellow'}>
                        {selectedNode?.status}
                      </Badge>
                    </Group>
                    <Text size="xs" c="dimmed">功能</Text>
                    <Text size="sm">{selectedNode?.function || selectedNode?.label}</Text>
                    <Text size="xs" c="dimmed">说明</Text>
                    <Text size="sm" c="dimmed">{selectedNode?.description || '等待节点执行后展示说明。'}</Text>
                    {selectedNodeOutput.llm_invoked && (
                      <Alert color="green" title="LLM 调用已完成">
                        <Text size="sm">
                          Agent：{selectedNodeOutput.llm_agent_name || selectedNodeOutput.agent_name}
                        </Text>
                        <Text size="sm">
                          模型：{selectedLlmCall.provider || 'unknown'} / {selectedLlmCall.model || 'unknown'}
                        </Text>
                        <Text size="sm">
                          返回字符数：{selectedLlmCall.raw_response_chars || 0}
                        </Text>
                        {selectedLlmCall.prompt_chars ? (
                          <Text size="sm">
                            Prompt 字符数：{selectedLlmCall.prompt_chars}
                          </Text>
                        ) : null}
                      </Alert>
                    )}
                    {selectedLlmPrompt && (
                      <>
                        <Text size="xs" c="dimmed">LLM 语句</Text>
                        <Code block>{selectedLlmPrompt}</Code>
                      </>
                    )}
                    <Divider />
                    <Text size="xs" c="dimmed">输入</Text>
                    <Code block>{stringify(selectedNode?.input)}</Code>
                    <Text size="xs" c="dimmed">输出</Text>
                    <Code block>{stringify(selectedNode?.output)}</Code>
                  </Stack>
                </ScrollArea>
              </Tabs.Panel>

              <Tabs.Panel value="run" pt="md">
                <ScrollArea h={420}>
                  <Code block>{stringify(run || { payload: buildPayload() })}</Code>
                </ScrollArea>
              </Tabs.Panel>
            </Tabs>
              </Paper>
            </Grid.Col>
          </Grid>
        </Stack>
      </Paper>

      {!isChapterCheck && (
        <Paper p="md" withBorder>
          <Stack gap="sm">
            <Title order={4}>用户处理</Title>
            <Textarea label="对话反馈/修改意见" minRows={feedbackRows} autosize value={feedback} onChange={(event) => setFeedback(event.currentTarget.value)} />
            <Group justify="space-between" align="flex-end">
              <Box w={260} maw="100%">
                <Select
                  label="修改模式"
                  data={revisionOptions}
                  value={revisionMode}
                  onChange={(value) => setRevisionMode((value as RevisionMode) || 'partial_rewrite')}
                  allowDeselect={false}
                />
              </Box>
              <Group gap="sm">
                <Button leftSection={<IconEdit size={16} />} variant="light" loading={submitting} disabled={!run || run.committed} onClick={() => respondWorkflow('request_changes')}>
                  提交修改
                </Button>
                <Button leftSection={<IconCircleCheck size={16} />} color="green" loading={submitting} disabled={!run || run.status !== 'waiting_human'} onClick={() => respondWorkflow('approve')}>
                  批准写库
                </Button>
                <Button color="red" variant="subtle" loading={submitting} disabled={!run || run.committed} onClick={() => respondWorkflow('reject')}>
                  中止
                </Button>
              </Group>
            </Group>
          </Stack>
        </Paper>
      )}

      <Paper p="md" withBorder>
        <Group gap="xs">
          <IconSend size={16} />
          <Text size="sm" c="dimmed">
            该页面是 {typeLabel[agentType]} 的独立工作流页。启动后节点、对话、审查和{isChapterCheck ? '检查结果' : '写库结果'}全部来自真实 `/api/hierarchy-agent/*` 接口。
          </Text>
        </Group>
      </Paper>
    </Stack>
  );
};

export const WorldWorkflow: React.FC = () => <HierarchyWorkflow fixedType="world" />;
export const WorldviewWorkflow: React.FC = () => <HierarchyWorkflow fixedType="worldview" />;
export const NovelWorkflow: React.FC = () => <HierarchyWorkflow fixedType="novel" />;
export const OutlineWorkflow: React.FC = () => <HierarchyWorkflow fixedType="outline" />;
export const ChapterWorkflow: React.FC = () => <HierarchyWorkflow fixedType="chapter" />;
