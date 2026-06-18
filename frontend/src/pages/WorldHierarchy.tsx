import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Alert,
  Badge,
  Box,
  Button,
  Checkbox,
  Group,
  Loader,
  Modal,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import {
  IconBook,
  IconEdit,
  IconGitBranch,
  IconMessageCircle,
  IconPlayerPlay,
  IconRefresh,
  IconTrash,
  IconWorld,
} from '@tabler/icons-react';
import { api } from '../api/client';
import { getWorldviewDisplayName } from '../utils/worldview';

type NodeKind = 'root' | 'world' | 'worldview' | 'novel' | 'outline' | 'chapter';
type AgentAction = 'create' | 'update' | 'delete';
type RevisionMode = 'partial_rewrite' | 'full_rewrite' | 'content_rewrite';

type ChapterNode = {
  id: string;
  name: string;
  content?: string;
  outline_id?: string;
  novel_id?: string;
  worldview_id?: string;
  world_id?: string;
};

type OutlineNode = {
  outline_id: string;
  novel_id?: string;
  worldview_id?: string;
  world_id: string;
  title: string;
  summary?: string;
  chapters: ChapterNode[];
};

type NovelNode = {
  novel_id: string;
  world_id: string;
  name: string;
  summary?: string;
  outlines: OutlineNode[];
};

type WorldviewNode = {
  worldview_id: string;
  world_id: string;
  name?: string;
  title?: string;
  summary?: string;
};

type WorldNode = {
  world_id: string;
  name: string;
  summary?: string;
  worldviews?: WorldviewNode[];
  novels?: NovelNode[];
};

const uniqueWorldsById = (items: WorldNode[]): WorldNode[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (!item.world_id || seen.has(item.world_id)) return false;
    seen.add(item.world_id);
    return true;
  });
};

type SelectedNode = {
  kind: NodeKind;
  id: string;
  label: string;
  data?: WorldNode | WorldviewNode | NovelNode | OutlineNode | ChapterNode;
};

type WorkflowNode = {
  node_id: string;
  label: string;
  status: string;
  input: unknown;
  output: unknown;
  started_at?: string;
  completed_at?: string;
};

type AgentRun = {
  run_id: string;
  agent_type: NodeKind;
  agent_name: string;
  action: AgentAction;
  status: string;
  iterations: number;
  review_required: boolean;
  pending_payload: Record<string, unknown>;
  commit_result?: Record<string, unknown>;
  created_at?: string;
  nodes: WorkflowNode[];
};

type AgentForm = {
  action: AgentAction;
  entity: Exclude<NodeKind, 'root'>;
  name: string;
  summary: string;
  content: string;
  message: string;
  feedback: string;
  revisionMode: RevisionMode;
  worldviewId: string;
  cascade: boolean;
};

type EntityListRow = {
  kind: 'world';
  id: string;
  label: string;
  summary: string;
  parent: string;
  data: WorldNode;
};

const emptyAgentForm: AgentForm = {
  action: 'create',
  entity: 'world',
  name: '',
  summary: '',
  content: '',
  message: '',
  feedback: '',
  revisionMode: 'partial_rewrite',
  worldviewId: '',
  cascade: false,
};

const labels: Record<NodeKind, string> = {
  root: '全部世界',
  world: '世界',
  worldview: '世界观',
  novel: '小说',
  outline: '大纲',
  chapter: '章节',
};

const statusColor: Record<string, string> = {
  completed: '#2f9e44',
  waiting: '#f08c00',
  blocked: '#c92a2a',
  failed: '#c92a2a',
  skipped: '#868e96',
};

const actionLabel: Record<AgentAction, string> = {
  create: '创建',
  update: '修改',
  delete: '删除',
};

const revisionModeOptions: Array<{ value: RevisionMode; label: string }> = [
  { value: 'partial_rewrite', label: '指定局部重写' },
  { value: 'content_rewrite', label: '指定内容重写' },
  { value: 'full_rewrite', label: '完全重写' },
];

const stringify = (value: unknown) => JSON.stringify(value ?? {}, null, 2);

export const WorldHierarchy: React.FC = () => {
  const navigate = useNavigate();
  const [worlds, setWorlds] = useState<WorldNode[]>([]);
  const [selected, setSelected] = useState<SelectedNode>({ kind: 'root', id: 'root', label: labels.root });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentOpen, setAgentOpen] = useState(false);
  const [agentForm, setAgentForm] = useState<AgentForm>(emptyAgentForm);
  const [activeRun, setActiveRun] = useState<AgentRun | null>(null);
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [selectedWorkflowNode, setSelectedWorkflowNode] = useState<WorkflowNode | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const loadWorlds = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const worldResponse = await api.listWorlds();
      const nextWorlds = uniqueWorldsById(worldResponse.data || []);
      setWorlds(nextWorlds);
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadRuns = useCallback(async () => {
    try {
      const agentTypes: Array<Exclude<NodeKind, 'root'>> = ['world', 'worldview', 'novel', 'outline', 'chapter'];
      const responses = await Promise.all(
        agentTypes.map((agentType) => api.listHierarchyAgents({ agent_type: agentType, page: 1, page_size: 10 })),
      );
      const nextRuns = responses
        .flatMap((response) => response.data?.runs || [])
        .sort((left, right) => String(right.created_at || '').localeCompare(String(left.created_at || '')))
        .slice(0, 50);
      setRuns(nextRuns);
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
    }
  }, []);

  useEffect(() => {
    loadWorlds();
    loadRuns();
  }, [loadWorlds, loadRuns]);

  const worldById = useMemo(() => new Map(worlds.map((world) => [world.world_id, world])), [worlds]);

  const owningWorld = useMemo(() => {
    if (selected.kind === 'world') return selected.data as WorldNode;
    if (selected.kind === 'worldview') return worldById.get((selected.data as WorldviewNode).world_id);
    if (selected.kind === 'novel') return worldById.get((selected.data as NovelNode).world_id);
    if (selected.kind === 'outline') return worldById.get((selected.data as OutlineNode).world_id);
    if (selected.kind === 'chapter') return worldById.get((selected.data as ChapterNode).world_id || '');
    return undefined;
  }, [selected, worldById]);

  const worldRows = useMemo(() => (
    worlds.map((world) => ({
        kind: 'world',
        id: world.world_id,
        label: world.name,
        summary: world.summary || '',
        parent: '',
        data: world,
      } as EntityListRow))
  ), [worlds]);

  const workflowGraph = useMemo(() => {
    const sourceNodes = activeRun?.nodes || [];
    const nodes: Node[] = sourceNodes.map((item, index) => ({
      id: `${item.node_id}-${index}`,
      position: { x: 40 + index * 210, y: 80 },
      data: { label: item.label },
      style: {
        width: 170,
        minHeight: 64,
        borderRadius: 8,
        border: `1px solid ${statusColor[item.status] || '#228be6'}`,
        background: '#101827',
        color: '#f8f9fa',
        fontSize: 12,
      },
    }));
    const edges: Edge[] = nodes.slice(1).map((node, index) => ({
      id: `edge-${index}`,
      source: nodes[index].id,
      target: node.id,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { stroke: '#4dabf7' },
    }));
    return { nodes, edges };
  }, [activeRun]);

  const selectNode = (kind: NodeKind, id: string, label: string, data: SelectedNode['data']) => {
    setSelected({ kind, id, label, data });
  };

  const childActions = useMemo(() => ['world'] as const, []);

  const openAgentDialog = (action: AgentAction, entity: Exclude<NodeKind, 'root'>) => {
    if (action !== 'delete') {
      const current = selected.data as WorldNode | undefined;
      const params = new URLSearchParams({
        type: entity,
        action,
        world_id: current?.world_id || worlds[0]?.world_id || 'world_default',
      });
      if (action === 'update' && current?.world_id) {
        params.set('id', current.world_id);
        params.set('name', current.name || '');
        params.set('summary', current.summary || '');
      }
      navigate(`/workflow/${entity}?${params.toString()}`);
      return;
    }
    const current = selected.data as any;
    const nextForm: AgentForm = {
      ...emptyAgentForm,
      action,
      entity,
      name: current?.name || current?.title || selected.label,
      summary: current?.summary || '',
      content: current?.content || '',
      message: `${actionLabel[action]}${labels[entity]}`,
      worldviewId: entity === 'outline' ? (current?.worldview_id || owningWorld?.worldviews?.[0]?.worldview_id || '') : '',
    };
    setAgentForm(nextForm);
    setActiveRun(null);
    setSelectedWorkflowNode(null);
    setAgentOpen(true);
  };

  const buildPayload = (): Record<string, unknown> => {
    const base: Record<string, unknown> = {};
    if (agentForm.action !== 'delete') {
      base.name = agentForm.name;
      if (agentForm.entity === 'chapter') {
        base.content = agentForm.content || agentForm.summary;
      } else {
        base.summary = agentForm.summary;
      }
    }
    if (agentForm.action === 'update' || agentForm.action === 'delete') {
      base.target_id = selected.id;
    }
    if (agentForm.action === 'delete') {
      base.cascade = agentForm.cascade;
    }
    if (agentForm.action === 'create') {
      if (agentForm.entity === 'worldview' || agentForm.entity === 'novel') {
        base.world_id = (selected.data as WorldNode).world_id;
      }
      if (agentForm.entity === 'outline') {
        const novel = selected.data as NovelNode;
        base.novel_id = novel.novel_id;
        base.world_id = novel.world_id;
        if (agentForm.worldviewId) base.worldview_id = agentForm.worldviewId;
      }
      if (agentForm.entity === 'chapter') {
        const outline = selected.data as OutlineNode;
        base.outline_id = outline.outline_id;
      }
    }
    return base;
  };

  const startWorkflow = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const response = await api.startHierarchyAgent({
        agent_type: agentForm.entity,
        action: agentForm.action,
        message: agentForm.message,
        payload: buildPayload(),
      });
      const run = response.data.run as AgentRun;
      setActiveRun(run);
      setSelectedWorkflowNode(run.nodes[run.nodes.length - 1] || null);
      await loadRuns();
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const respondWorkflow = async (decision: 'approve' | 'request_changes' | 'reject') => {
    if (!activeRun) return;
    setSubmitting(true);
    setError(null);
    try {
      const response = await api.respondHierarchyAgent({
        run_id: activeRun.run_id,
        decision,
        message: agentForm.feedback,
        revision_mode: decision === 'request_changes' ? agentForm.revisionMode : undefined,
        manual_edit: decision === 'request_changes',
        payload: decision === 'request_changes' ? buildPayload() : undefined,
      });
      const run = response.data.run as AgentRun;
      setActiveRun(run);
      setSelectedWorkflowNode(run.nodes[run.nodes.length - 1] || null);
      await loadRuns();
      if (run.status === 'completed') {
        await loadWorlds();
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Stack gap="md" style={{ height: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Title order={2}>世界管理 Agent 工作台</Title>
          <Text size="sm" c="dimmed">只显示世界列表；世界观、小说、大纲、章节不在此页展开。</Text>
        </Box>
        <Group gap="xs">
          <Button leftSection={<IconRefresh size={16} />} variant="light" onClick={() => { loadWorlds(); loadRuns(); }}>刷新</Button>
          {childActions.map((kind) => (
            <Button key={kind} leftSection={<IconPlayerPlay size={16} />} onClick={() => openAgentDialog('create', kind)}>
              创建{labels[kind]}
            </Button>
          ))}
          {selected.kind !== 'root' && (
            <>
              <Button leftSection={<IconEdit size={16} />} variant="light" onClick={() => openAgentDialog('update', selected.kind as Exclude<NodeKind, 'root'>)}>修改</Button>
              <Button leftSection={<IconTrash size={16} />} color="red" variant="light" onClick={() => openAgentDialog('delete', selected.kind as Exclude<NodeKind, 'root'>)}>删除</Button>
            </>
          )}
        </Group>
      </Group>

      {error && <Alert color="red">{error}</Alert>}

      <Box style={{ display: 'grid', gridTemplateColumns: 'minmax(520px, 1fr) 360px', gap: 12, minHeight: 0, flex: 1 }}>
        <Paper p="sm" withBorder style={{ overflow: 'hidden' }}>
          <Group justify="space-between" mb="sm">
            <Box>
              <Text fw={800}>世界列表</Text>
              <Badge variant="light">真实数据库查询</Badge>
            </Box>
            <Group gap="xs">
              {loading && <Loader size="sm" />}
              <Text size="xs" c="dimmed">共 {worldRows.length} 条</Text>
            </Group>
          </Group>
          <ScrollArea h="calc(100vh - 205px)">
            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>类型</Table.Th>
                  <Table.Th>ID</Table.Th>
                  <Table.Th>名称</Table.Th>
                  <Table.Th>摘要</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {worldRows.map((item) => (
                    <Table.Tr key={`${item.kind}:${item.id}`} onClick={() => navigate(`/worlds/${encodeURIComponent(item.id)}`)} style={{ cursor: 'pointer' }} title="点击进入世界详情">
                      <Table.Td><Badge size="sm">{labels[item.kind]}</Badge></Table.Td>
                      <Table.Td><Text size="xs" truncate maw={160}>{item.id}</Text></Table.Td>
                      <Table.Td>{item.label}</Table.Td>
                      <Table.Td><Text size="sm" truncate maw={520}>{item.summary || ''}</Text></Table.Td>
                    </Table.Tr>
                  ))}
                {worldRows.length === 0 && (
                  <Table.Tr>
                    <Table.Td colSpan={4}>
                      <Text ta="center" c="dimmed" py="xl">当前没有世界记录</Text>
                    </Table.Td>
                  </Table.Tr>
                )}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        </Paper>

        <Paper p="sm" withBorder style={{ overflow: 'hidden' }}>
          <Group justify="space-between" mb="xs">
            <Text fw={800}>最近工作流</Text>
            <IconGitBranch size={18} />
          </Group>
          <ScrollArea h="calc(100vh - 190px)">
            <Stack gap="xs">
              {runs.map((run) => (
                <Paper key={run.run_id} p="xs" withBorder style={{ cursor: 'pointer' }} onClick={() => { setActiveRun(run); setAgentOpen(true); }}>
                  <Group justify="space-between" gap="xs">
                    <Text size="sm" fw={700}>{run.agent_name}</Text>
                    <Badge color={run.status === 'completed' ? 'green' : run.status === 'review_failed' ? 'red' : 'yellow'}>{run.status}</Badge>
                  </Group>
                  <Text size="xs" c="dimmed">{actionLabel[run.action]} / 迭代 {run.iterations}</Text>
                </Paper>
              ))}
            </Stack>
          </ScrollArea>
        </Paper>
      </Box>

      <Modal opened={agentOpen} onClose={() => setAgentOpen(false)} title="Agent 对话与工作流" size="95%">
        <Box style={{ display: 'grid', gridTemplateColumns: '360px minmax(420px, 1fr) 360px', gap: 12, minHeight: 620 }}>
          <Paper p="md" withBorder>
            <Stack gap="sm">
              <Group>
                <Badge>{labels[agentForm.entity]}</Badge>
                <Badge variant="light">{actionLabel[agentForm.action]}</Badge>
                {agentForm.entity !== 'world' && <Badge color="orange">需要审查</Badge>}
              </Group>
              {agentForm.action !== 'delete' && (
                <>
                  <TextInput label="名称" value={agentForm.name} onChange={(event) => setAgentForm({ ...agentForm, name: event.currentTarget.value })} />
                  {agentForm.entity === 'chapter' ? (
                    <Textarea label="章节内容" minRows={5} value={agentForm.content} onChange={(event) => setAgentForm({ ...agentForm, content: event.currentTarget.value })} />
                  ) : (
                    <Textarea label="摘要/设定" minRows={4} value={agentForm.summary} onChange={(event) => setAgentForm({ ...agentForm, summary: event.currentTarget.value })} />
                  )}
                  {agentForm.entity === 'outline' && agentForm.action === 'create' && (
                    <Select
                      label="关联世界观"
                      clearable
                      data={(owningWorld?.worldviews || []).map((worldview) => ({ value: worldview.worldview_id, label: getWorldviewDisplayName(worldview) }))}
                      value={agentForm.worldviewId || null}
                      onChange={(value) => setAgentForm({ ...agentForm, worldviewId: value || '' })}
                    />
                  )}
                </>
              )}
              {agentForm.action === 'delete' && (
                <Checkbox
                  label="cascade=true 级联删除"
                  checked={agentForm.cascade}
                  onChange={(event) => setAgentForm({ ...agentForm, cascade: event.currentTarget.checked })}
                />
              )}
              <Textarea label="对 Agent 的消息" minRows={3} value={agentForm.message} onChange={(event) => setAgentForm({ ...agentForm, message: event.currentTarget.value })} />
              <Button leftSection={<IconMessageCircle size={16} />} loading={submitting} onClick={startWorkflow} disabled={Boolean(activeRun && activeRun.status !== 'rejected')}>
                启动独立 Agent
              </Button>
              <Textarea label="人工反馈/修改意见" minRows={3} value={agentForm.feedback} onChange={(event) => setAgentForm({ ...agentForm, feedback: event.currentTarget.value })} />
              <Select
                label="修改模式"
                description="默认局部重写；小部分修改只改命名或少量字段，完全重写才允许大范围重构。"
                data={revisionModeOptions}
                value={agentForm.revisionMode}
                onChange={(value) => setAgentForm({ ...agentForm, revisionMode: (value as RevisionMode) || 'partial_rewrite' })}
              />
              <Alert color="blue">
                表单中的名称、摘要、内容会作为人工手动修改提交；Agent 必须按所选修改模式限制改动范围。
              </Alert>
              <Group grow>
                <Button variant="light" loading={submitting} disabled={!activeRun} onClick={() => respondWorkflow('request_changes')}>提交手动修改</Button>
                <Button color="green" loading={submitting} disabled={!activeRun || activeRun.status !== 'waiting_human'} onClick={() => respondWorkflow('approve')}>批准写入</Button>
              </Group>
              <Button color="red" variant="subtle" disabled={!activeRun || activeRun.status === 'completed'} onClick={() => respondWorkflow('reject')}>终止工作流</Button>
            </Stack>
          </Paper>

          <Paper withBorder style={{ overflow: 'hidden' }}>
            <ReactFlow
              nodes={workflowGraph.nodes}
              edges={workflowGraph.edges}
              onNodeClick={(_, node) => {
                const index = workflowGraph.nodes.findIndex((item) => item.id === node.id);
                setSelectedWorkflowNode(activeRun?.nodes[index] || null);
              }}
              fitView
            >
              <Background />
              <Controls />
              <MiniMap />
            </ReactFlow>
          </Paper>

          <Paper p="md" withBorder style={{ overflow: 'hidden' }}>
            <ScrollArea h={590}>
              <Stack gap="sm">
                <Text fw={800}>节点输入/输出</Text>
                {activeRun && (
                  <Alert color={activeRun.status === 'completed' ? 'green' : activeRun.status === 'review_failed' ? 'red' : 'yellow'}>
                    {activeRun.agent_name} / {activeRun.status} / 第 {activeRun.iterations} 轮
                  </Alert>
                )}
                {selectedWorkflowNode ? (
                  <>
                    <Group justify="space-between">
                      <Text fw={700}>{selectedWorkflowNode.label}</Text>
                      <Badge>{selectedWorkflowNode.status}</Badge>
                    </Group>
                    <Text size="xs" c="dimmed">输入</Text>
                    <Box component="pre" style={{ whiteSpace: 'pre-wrap', fontSize: 12, background: '#0b1020', padding: 10, borderRadius: 6 }}>{stringify(selectedWorkflowNode.input)}</Box>
                    <Text size="xs" c="dimmed">输出</Text>
                    <Box component="pre" style={{ whiteSpace: 'pre-wrap', fontSize: 12, background: '#0b1020', padding: 10, borderRadius: 6 }}>{stringify(selectedWorkflowNode.output)}</Box>
                  </>
                ) : (
                  <Text size="sm" c="dimmed">启动或选择一个工作流节点后查看真实输入与输出。</Text>
                )}
              </Stack>
            </ScrollArea>
          </Paper>
        </Box>
      </Modal>
    </Stack>
  );
};
