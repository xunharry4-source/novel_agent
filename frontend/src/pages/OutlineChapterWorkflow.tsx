import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Node,
  Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Alert,
  Badge,
  Button,
  Divider,
  FileInput,
  Grid,
  Group,
  JsonInput,
  Modal,
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
  IconAlertTriangle,
  IconEdit,
  IconFileImport,
  IconPlayerPlay,
  IconRefresh,
  IconTrash,
} from '@tabler/icons-react';
import { api, apiClient } from '../api/client';
import { getWorldviewDisplayName } from '../utils/worldview';

type Outline = {
  outline_id: string;
  world_id: string;
  novel_id?: string;
  worldview_id: string;
  title: string;
  summary?: string;
  timestamp?: string;
};

type LoreItem = {
  id: string;
  type: string;
  name: string;
  content?: string;
  outline_id?: string;
  world_id?: string;
  novel_id?: string;
  worldview_id?: string;
  timestamp?: string;
};

type World = {
  world_id: string;
  name: string;
  summary?: string;
};

type Worldview = {
  worldview_id: string;
  world_id: string;
  name?: string;
  title?: string;
  summary?: string;
};

type Novel = {
  novel_id: string;
  world_id: string;
  name: string;
  summary?: string;
};

type WorkflowLog = {
  id: string;
  level: 'info' | 'success' | 'error';
  message: string;
  payload?: unknown;
};

type HumanMode = 'delete-outline' | 'delete-chapter';

const toOptions = <T extends Record<string, string | undefined>>(items: T[], valueKey: keyof T, labelKey: keyof T) =>
  items.map((item) => ({ value: String(item[valueKey] || ''), label: String(item[labelKey] || '') }));

const toWorldviewOptions = (items: Worldview[]) =>
  items.map((item) => ({ value: item.worldview_id, label: getWorldviewDisplayName(item) }));

const uniqueWorldsById = (items: World[]): World[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (!item.world_id || seen.has(item.world_id)) return false;
    seen.add(item.world_id);
    return true;
  });
};

export const OutlineChapterWorkflow: React.FC = () => {
  const navigate = useNavigate();
  const [worlds, setWorlds] = useState<World[]>([]);
  const [worldviews, setWorldviews] = useState<Worldview[]>([]);
  const [novels, setNovels] = useState<Novel[]>([]);
  const [outlines, setOutlines] = useState<Outline[]>([]);
  const [chapters, setChapters] = useState<LoreItem[]>([]);
  const [selectedWorldId, setSelectedWorldId] = useState('');
  const [selectedWorldview, setSelectedWorldview] = useState('');
  const [selectedNovel, setSelectedNovel] = useState<string | null>(null);
  const [selectedOutline, setSelectedOutline] = useState<string | null>(null);
  const [selectedChapter, setSelectedChapter] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState<WorkflowLog[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<unknown>(null);
  const [humanMode, setHumanMode] = useState<HumanMode | null>(null);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importWorldviewId, setImportWorldviewId] = useState('');
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importing, setImporting] = useState(false);

  const [outlineForm, setOutlineForm] = useState({
    name: '',
    summary: '',
    updatedName: '',
    updatedContent: '',
  });
  const [chapterForm, setChapterForm] = useState({
    id: '',
    title: '',
    content: '',
    updatedTitle: '',
    updatedContent: '',
  });
  const [humanForm, setHumanForm] = useState({
    deleteReason: '',
  });

  const addLog = useCallback((log: Omit<WorkflowLog, 'id'>) => {
    setLogs((current) => [{ id: `${Date.now()}-${Math.random()}`, ...log }, ...current].slice(0, 80));
  }, []);

  const loadWorlds = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.listWorlds();
      const nextWorlds = uniqueWorldsById((response.data as World[]) || []);
      setWorlds(nextWorlds);
      if (nextWorlds.length > 0) {
        setSelectedWorldId((current) => current || nextWorlds[0].world_id);
      } else {
        setSelectedWorldId('');
        setSelectedWorldview('');
        setSelectedNovel(null);
        setSelectedOutline(null);
        setSelectedChapter(null);
        setWorldviews([]);
        setNovels([]);
        setOutlines([]);
        setChapters([]);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setWorlds([]);
      setWorldviews([]);
      setNovels([]);
      setOutlines([]);
      setChapters([]);
      addLog({ level: 'error', message: '加载世界列表失败', payload: message });
    } finally {
      setLoading(false);
    }
  }, [addLog]);

  const loadState = useCallback(async () => {
    if (!selectedWorldId) {
      setWorldviews([]);
      setNovels([]);
      setOutlines([]);
      setChapters([]);
      setSelectedWorldview('');
      setSelectedNovel(null);
      setSelectedOutline(null);
      setSelectedChapter(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const stateRequest = selectedWorldview
        ? api.getOutlineChapterState({
          world_id: selectedWorldId,
          worldview_id: selectedWorldview,
          outline_id: selectedOutline || undefined,
          page: 1,
          page_size: 50,
        })
        : Promise.resolve({ data: { status: 'success', world_id: selectedWorldId, chapters: [] } });

      const [worldviewRes, novelRes, outlineRes, stateRes] = await Promise.all([
        api.listWorldviews({ world_id: selectedWorldId, page: 1, page_size: 50 }),
        api.listNovels({ world_id: selectedWorldId, page: 1, page_size: 50 }),
        api.listOutlines({ world_id: selectedWorldId, worldview_id: selectedWorldview || undefined, page: 1, page_size: 50 }),
        stateRequest,
      ]);
      const nextWorldviews = worldviewRes.data as Worldview[];
      const nextNovels = novelRes.data as Novel[];
      const nextOutlines = outlineRes.data as Outline[];
      const nextChapters = (stateRes.data?.chapters || []) as LoreItem[];
      const invalidWorldviews = nextWorldviews.filter((worldview) => worldview.world_id !== selectedWorldId);
      if (invalidWorldviews.length > 0) {
        throw new Error(`接口返回了非当前世界的世界观: ${invalidWorldviews.map((item) => item.worldview_id).join(', ')}`);
      }
      const invalidNovels = nextNovels.filter((novel) => novel.world_id !== selectedWorldId);
      if (invalidNovels.length > 0) {
        throw new Error(`接口返回了非当前世界的小说: ${invalidNovels.map((item) => item.novel_id).join(', ')}`);
      }
      const invalidOutlines = nextOutlines.filter((outline) => outline.world_id !== selectedWorldId);
      if (invalidOutlines.length > 0) {
        throw new Error(`接口返回了非当前世界的大纲: ${invalidOutlines.map((item) => item.outline_id).join(', ')}`);
      }
      const invalidChapters = nextChapters.filter((chapter) => chapter.world_id !== selectedWorldId);
      if (invalidChapters.length > 0) {
        throw new Error(`接口返回了非当前世界的章节: ${invalidChapters.map((item) => item.id).join(', ')}`);
      }
      setWorldviews(nextWorldviews);
      setNovels(nextNovels);
      setOutlines(nextOutlines);
      setChapters(nextChapters);
      setSelectedWorldview((current) => {
        if (nextWorldviews.some((worldview) => worldview.worldview_id === current)) return current;
        return nextWorldviews[0]?.worldview_id || '';
      });
      setSelectedOutline((current) => {
        if (nextOutlines.some((outline) => outline.outline_id === current)) return current;
        return nextOutlines[0]?.outline_id || null;
      });
      setSelectedNovel((current) => {
        if (nextNovels.some((novel) => novel.novel_id === current)) return current;
        const outlineNovel = nextOutlines.find((outline) => outline.outline_id === selectedOutline)?.novel_id;
        return outlineNovel || nextNovels[0]?.novel_id || null;
      });
      setSelectedChapter((current) => {
        if (nextChapters.some((chapter) => chapter.id === current)) return current;
        return null;
      });
      setLastResponse(stateRes.data);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '加载工作流状态失败', payload: message });
      setWorldviews([]);
      setNovels([]);
      setOutlines([]);
      setChapters([]);
    } finally {
      setLoading(false);
    }
  }, [addLog, selectedOutline, selectedWorldId, selectedWorldview]);

  useEffect(() => {
    loadWorlds();
  }, [loadWorlds]);

  useEffect(() => {
    if (selectedWorldId) {
      loadState();
    }
  }, [loadState, selectedWorldId]);

  const selectedWorldRecord = useMemo(
    () => worlds.find((world) => world.world_id === selectedWorldId),
    [selectedWorldId, worlds],
  );
  const selectedOutlineRecord = useMemo(
    () => outlines.find((outline) => outline.outline_id === selectedOutline),
    [outlines, selectedOutline],
  );
  const selectedWorldviewRecord = useMemo(
    () => worldviews.find((worldview) => worldview.worldview_id === selectedWorldview),
    [worldviews, selectedWorldview],
  );
  const selectedNovelRecord = useMemo(
    () => novels.find((novel) => novel.novel_id === selectedNovel),
    [novels, selectedNovel],
  );
  const selectedChapterRecord = useMemo(
    () => chapters.find((chapter) => chapter.id === selectedChapter),
    [chapters, selectedChapter],
  );

  const openWorkflow = useCallback((type: 'novel' | 'outline' | 'chapter', action: 'create' | 'update') => {
    const params = new URLSearchParams({
      type,
      action,
      world_id: selectedWorldId,
    });
    if (selectedWorldview) params.set('worldview_id', selectedWorldview);
    if (selectedNovel) params.set('novel_id', selectedNovel);
    if (selectedOutline) params.set('outline_id', selectedOutline);

    if (type === 'novel') {
      if (action === 'create') {
        params.set('name', outlineForm.name || 'Agent 新小说');
        params.set('summary', outlineForm.summary || '');
      } else if (selectedNovelRecord) {
        params.set('id', selectedNovelRecord.novel_id);
        params.set('name', outlineForm.updatedName || selectedNovelRecord.name || '');
        params.set('summary', outlineForm.updatedContent || selectedNovelRecord.summary || '');
      }
    }
    if (type === 'outline') {
      if (action === 'create') {
        params.set('name', outlineForm.name || 'Agent 新大纲');
        params.set('summary', outlineForm.summary || '');
      } else if (selectedOutlineRecord) {
        params.set('id', selectedOutlineRecord.outline_id);
        params.set('name', outlineForm.updatedName || selectedOutlineRecord.title || '');
        params.set('summary', outlineForm.updatedContent || selectedOutlineRecord.summary || '');
      }
    }
    if (type === 'chapter') {
      if (action === 'create') {
        params.set('name', chapterForm.title || 'Agent 新章节');
        params.set('content', chapterForm.content || '');
      } else if (selectedChapterRecord) {
        params.set('id', selectedChapterRecord.id);
        params.set('name', chapterForm.updatedTitle || selectedChapterRecord.name || '');
        params.set('content', chapterForm.updatedContent || selectedChapterRecord.content || '');
      }
    }
    navigate(`/workflow/${type}?${params.toString()}`);
  }, [
    chapterForm.content,
    chapterForm.title,
    chapterForm.updatedContent,
    chapterForm.updatedTitle,
    navigate,
    outlineForm.name,
    outlineForm.summary,
    outlineForm.updatedContent,
    outlineForm.updatedName,
    selectedChapterRecord,
    selectedNovel,
    selectedNovelRecord,
    selectedOutline,
    selectedOutlineRecord,
    selectedWorldId,
    selectedWorldview,
  ]);
  const dbGraph = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const worldNodeId = `world:${selectedWorldId || 'none'}`;
    nodes.push({
      id: worldNodeId,
      position: { x: -260, y: 180 },
      data: { label: selectedWorldRecord?.name || selectedWorldId || '必须选择世界' },
      type: 'input',
    });

    const worldviewNodeId = `worldview:${selectedWorldview || 'none'}`;
    nodes.push({
      id: worldviewNodeId,
      position: { x: 0, y: 180 },
      data: { label: selectedWorldviewRecord?.name || selectedWorldview },
    });
    if (selectedWorldId && selectedWorldview) {
      edges.push({
        id: `${worldNodeId}-${worldviewNodeId}`,
        source: worldNodeId,
        target: worldviewNodeId,
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    }

    const visibleOutlines = outlines.filter((outline) => outline.world_id === selectedWorldId && (!selectedWorldview || outline.worldview_id === selectedWorldview));
    visibleOutlines.forEach((outline, index) => {
      const outlineNodeId = `outline:${outline.outline_id}`;
      nodes.push({
        id: outlineNodeId,
        position: { x: 260, y: 70 + index * 120 },
        data: { label: outline.title },
        type: outline.outline_id === selectedOutline ? 'default' : undefined,
      });
      edges.push({
        id: `${worldviewNodeId}-${outlineNodeId}`,
        source: worldviewNodeId,
        target: outlineNodeId,
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    });

    chapters.forEach((chapter, index) => {
      const chapterNodeId = `chapter:${chapter.id}`;
      const sourceId = `outline:${chapter.outline_id || selectedOutline}`;
      nodes.push({
        id: chapterNodeId,
        position: { x: 560, y: 70 + index * 110 },
        data: { label: chapter.name },
        type: 'output',
      });
      edges.push({
        id: `${sourceId}-${chapterNodeId}`,
        source: sourceId,
        target: chapterNodeId,
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    });

    return { nodes, edges };
  }, [chapters, outlines, selectedOutline, selectedWorldId, selectedWorldRecord, selectedWorldview, selectedWorldviewRecord]);

  const verifyOutline = async (outlineId: string, expected?: { title?: string; summary?: string }) => {
    if (!selectedWorldId) throw new Error('查询验证失败: 必须选择世界');
    const response = await api.listOutlines({ world_id: selectedWorldId, outline_id: outlineId, page: 1, page_size: 10 });
    const nextOutlines = response.data as Outline[];
    setOutlines(nextOutlines);
    const found = nextOutlines.find((item) => item.outline_id === outlineId);
    if (!found) throw new Error(`查询验证失败: 大纲 ${outlineId} 不存在`);
    if (found.world_id !== selectedWorldId) {
      throw new Error(`查询验证失败: 大纲 ${outlineId} 不属于当前世界 ${selectedWorldId}`);
    }
    if (expected?.title && found.title !== expected.title) {
      throw new Error(`查询验证失败: 大纲标题未更新，实际 ${found.title}`);
    }
    if (expected?.summary && found.summary !== expected.summary) {
      throw new Error(`查询验证失败: 大纲内容未更新，实际 ${found.summary}`);
    }
    return found;
  };

  const verifyChapter = async (chapterId: string, expected?: { title?: string; content?: string; deleted?: boolean }) => {
    if (!selectedWorldId) throw new Error('查询验证失败: 必须选择世界');
    const response = await api.listLore({ world_id: selectedWorldId, outline_id: selectedOutline || undefined, worldview_id: selectedWorldview || undefined, page: 1, page_size: 50 });
    const items = response.data as LoreItem[];
    const nextChapters = items.filter((item) => item.type === 'prose');
    setChapters(nextChapters);
    const found = nextChapters.find((item) => item.id === chapterId);
    if (expected?.deleted) {
      if (found) throw new Error(`查询验证失败: 章节 ${chapterId} 删除后仍存在`);
      return null;
    }
    if (!found) throw new Error(`查询验证失败: 章节 ${chapterId} 不存在`);
    if (found.world_id !== selectedWorldId) {
      throw new Error(`查询验证失败: 章节 ${chapterId} 不属于当前世界 ${selectedWorldId}`);
    }
    if (expected?.title && found.name !== expected.title) {
      throw new Error(`查询验证失败: 章节标题未更新，实际 ${found.name}`);
    }
    if (expected?.content && found.content !== expected.content) {
      throw new Error('查询验证失败: 章节内容未更新为提交内容');
    }
    return found;
  };

  const createOutline = async () => {
    if (!selectedWorldId || !selectedWorldview || !outlineForm.name.trim()) {
      setError('创建大纲必须选择世界、世界观，并填写名称');
      return;
    }
    setError(null);
    try {
      const response = await api.createOutline({
        name: outlineForm.name,
        summary: outlineForm.summary,
        world_id: selectedWorldId,
        worldview_id: selectedWorldview,
      });
      const outlineId = response.data.outline_id;
      const verified = await verifyOutline(outlineId, { title: outlineForm.name });
      setSelectedOutline(outlineId);
      setLastResponse({ create: response.data, verify: verified });
      addLog({ level: 'success', message: `大纲创建并查询验证通过: ${outlineId}`, payload: verified });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '大纲创建失败', payload: message });
    }
  };

  const updateOutline = async () => {
    if (!selectedWorldId || !selectedOutline || !outlineForm.updatedName.trim() || !outlineForm.updatedContent.trim()) {
      setError('修改大纲必须选择世界、大纲，并填写标题与内容');
      return;
    }
    setError(null);
    try {
      const response = await api.updateArchiveItem({
        id: selectedOutline,
        type: 'outline',
        name: outlineForm.updatedName,
        content: outlineForm.updatedContent,
        world_id: selectedWorldId,
        worldview_id: selectedWorldview,
      });
      const verified = await verifyOutline(selectedOutline, {
        title: outlineForm.updatedName,
        summary: outlineForm.updatedContent,
      });
      setLastResponse({ update: response.data, verify: verified });
      addLog({ level: 'success', message: `大纲修改并查询验证通过: ${selectedOutline}`, payload: verified });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '大纲修改失败', payload: message });
    }
  };

  const deleteOutline = async () => {
    if (!selectedWorldId || !selectedOutline || !humanForm.deleteReason.trim()) {
      setError('删除大纲必须选择世界、大纲并填写删除原因');
      return;
    }
    setError(null);
    try {
      const response = await api.deleteArchiveItem({
        id: selectedOutline,
        type: 'outline',
        world_id: selectedWorldId,
        worldview_id: selectedWorldview,
      });
      const listResponse = await api.listOutlines({ world_id: selectedWorldId, worldview_id: selectedWorldview || undefined, page: 1, page_size: 50 });
      const nextOutlines = listResponse.data as Outline[];
      setOutlines(nextOutlines);
      if (nextOutlines.some((item) => item.outline_id === selectedOutline)) {
        throw new Error(`查询验证失败: 大纲 ${selectedOutline} 删除后仍存在`);
      }
      addLog({
        level: 'success',
        message: `大纲删除并查询验证通过: ${selectedOutline}`,
        payload: { delete: response.data, reason: humanForm.deleteReason },
      });
      setLastResponse({ delete: response.data, reason: humanForm.deleteReason });
      setSelectedOutline(nextOutlines[0]?.outline_id || null);
      setHumanMode(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '大纲删除失败', payload: message });
    }
  };

  const saveChapter = async () => {
    if (!selectedWorldId || !selectedOutline || !chapterForm.title.trim() || !chapterForm.content.trim()) {
      setError('保存章节必须选择世界、大纲，并填写章节标题与内容');
      return;
    }
    const chapterId = chapterForm.id.trim() || `chapter_${Date.now()}`;
    setError(null);
    try {
      const response = await api.updateArchiveItem({
        id: chapterId,
        type: 'prose',
        name: chapterForm.title,
        content: chapterForm.content,
        world_id: selectedWorldId,
        outline_id: selectedOutline,
        worldview_id: selectedWorldview,
      });
      const verified = await verifyChapter(chapterId, {
        title: chapterForm.title,
        content: chapterForm.content,
      });
      setSelectedChapter(chapterId);
      setChapterForm((current) => ({ ...current, id: chapterId }));
      setLastResponse({ save: response.data, verify: verified });
      addLog({ level: 'success', message: `章节保存并查询验证通过: ${chapterId}`, payload: verified });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '章节保存失败', payload: message });
    }
  };

  const updateChapter = async () => {
    const chapterId = selectedChapter || chapterForm.id;
    if (!selectedWorldId || !chapterId || !chapterForm.updatedTitle.trim() || !chapterForm.updatedContent.trim()) {
      setError('修改章节必须选择世界、章节，并填写标题与内容');
      return;
    }
    setError(null);
    try {
      const response = await api.updateArchiveItem({
        id: chapterId,
        type: 'prose',
        name: chapterForm.updatedTitle,
        content: chapterForm.updatedContent,
        world_id: selectedWorldId,
        outline_id: selectedOutline || undefined,
        worldview_id: selectedWorldview,
      });
      const verified = await verifyChapter(chapterId, {
        title: chapterForm.updatedTitle,
        content: chapterForm.updatedContent,
      });
      setLastResponse({ update: response.data, verify: verified });
      addLog({ level: 'success', message: `章节修改并查询验证通过: ${chapterId}`, payload: verified });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '章节修改失败', payload: message });
    }
  };

  const deleteChapter = async () => {
    if (!selectedWorldId || !selectedChapter || !humanForm.deleteReason.trim()) {
      setError('删除章节必须选择世界、章节并填写删除原因');
      return;
    }
    setError(null);
    try {
      const response = await api.deleteArchiveItem({
        id: selectedChapter,
        type: 'prose',
        world_id: selectedWorldId,
        outline_id: selectedOutline || undefined,
        worldview_id: selectedWorldview,
      });
      await verifyChapter(selectedChapter, { deleted: true });
      addLog({
        level: 'success',
        message: `章节删除并查询验证通过: ${selectedChapter}`,
        payload: { delete: response.data, reason: humanForm.deleteReason },
      });
      setLastResponse({ delete: response.data, reason: humanForm.deleteReason });
      setSelectedChapter(null);
      setHumanMode(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '章节删除失败', payload: message });
    }
  };

  const openImportModal = () => {
    setImportWorldviewId(selectedWorldview || worldviews[0]?.worldview_id || '');
    setImportFile(null);
    setImportModalOpen(true);
  };

  const importWorldviewHierarchy = async () => {
    if (!selectedWorldId || !importWorldviewId || !importFile) {
      setError('导入世界观必须选择世界、世界观，并选择 json/md/xml/opml 文件');
      return;
    }
    setImporting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('world_id', selectedWorldId);
      form.append('worldview_id', importWorldviewId);
      form.append('file', importFile);
      const response = await fetch(`${apiClient.defaults.baseURL}/api/worldviews/import`, {
        method: 'POST',
        body: form,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(`导入失败 ${response.status}: ${JSON.stringify(data)}`);
      }
      const importedEntries = (data.entries || []) as Array<{ id: string; name: string; path: string }>;
      if (importedEntries.length === 0) {
        throw new Error('导入接口未返回任何写入条目，拒绝伪造成功');
      }
      const verifyTarget = importedEntries[0];
      const verifyResponse = await api.listLore({
        world_id: selectedWorldId,
        worldview_id: importWorldviewId,
        query: verifyTarget.name,
        page: 1,
        page_size: 50,
      });
      const verifiedItems = verifyResponse.data as LoreItem[];
      const found = verifiedItems.find((item) => item.id === verifyTarget.id && item.world_id === selectedWorldId && item.worldview_id === importWorldviewId);
      if (!found) {
        throw new Error(`导入后查询验证失败: ${verifyTarget.id} 未在当前世界/世界观中找到`);
      }
      setSelectedWorldview(importWorldviewId);
      setImportModalOpen(false);
      setImportFile(null);
      setLastResponse({ import: data, verify: found });
      addLog({ level: 'success', message: `世界观层级导入并查询验证通过: ${data.imported_count} 条`, payload: { import: data, verify: found } });
      await loadState();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      addLog({ level: 'error', message: '世界观导入失败', payload: message });
    } finally {
      setImporting(false);
    }
  };

  return (
    <Stack gap="md">
      <Group justify="space-between" align="flex-end">
        <div>
          <Title order={2}>大纲与章节迭代工作流</Title>
          <Text c="dimmed" size="sm">React Flow 展示流程，Mantine 表单执行人工审核；所有写操作后立即查询验证。</Text>
        </div>
        <Group>
          <Badge color="cyan" variant="light">API {String(apiClient.defaults.baseURL)}</Badge>
          <Button leftSection={<IconFileImport size={16} />} variant="light" disabled={!selectedWorldId || worldviews.length === 0} onClick={openImportModal}>导入世界观</Button>
          <Button leftSection={<IconRefresh size={16} />} variant="light" loading={loading} onClick={loadState} disabled={!selectedWorldId}>刷新</Button>
        </Group>
      </Group>

      {!selectedWorldId && (
        <Alert color="yellow" icon={<IconAlertTriangle size={18} />} title="必须选择世界">
          当前没有可用世界，或世界列表尚未加载完成。
        </Alert>
      )}

      {error && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} title="错误未被隐藏">
          {error}
        </Alert>
      )}

      <Grid gutter="md">
        <Grid.Col span={{ base: 12, lg: 3 }}>
          <Stack gap="md">
            <Paper p="md" radius="sm" withBorder>
              <Stack gap="sm">
                <Title order={4}>项目上下文</Title>
                <Select
                  required
                  label="世界"
                  placeholder="必须选择世界"
                  data={worlds.map((world) => ({
                    value: world.world_id,
                    label: `${world.name} (${world.world_id})`,
                  }))}
                  value={selectedWorldId || null}
                  onChange={(value) => {
                    setSelectedWorldId(value || '');
                    setSelectedWorldview('');
                    setSelectedNovel(null);
                    setSelectedOutline(null);
                    setSelectedChapter(null);
                    setOutlines([]);
                    setChapters([]);
                  }}
                  allowDeselect={false}
                  searchable
                />
                <Select
                  label="世界观"
                  data={toWorldviewOptions(worldviews)}
                  value={selectedWorldview || null}
                  onChange={(value) => {
                    setSelectedWorldview(value || '');
                    setSelectedOutline(null);
                    setSelectedChapter(null);
                    setChapters([]);
                  }}
                  disabled={!selectedWorldId}
                  searchable
                />
                <Select
                  label="小说"
                  data={novels.map((novel) => ({
                    value: novel.novel_id,
                    label: `${novel.name} (${novel.novel_id})`,
                  }))}
                  value={selectedNovel}
                  onChange={(value) => {
                    setSelectedNovel(value);
                    setSelectedOutline(null);
                    setSelectedChapter(null);
                  }}
                  disabled={!selectedWorldId}
                  searchable
                  clearable
                />
                <Select
                  label="小说大纲/项目"
                  data={outlines.map((outline) => ({
                    value: outline.outline_id,
                    label: `${outline.title} (${outline.outline_id})`,
                  }))}
                  value={selectedOutline}
                  onChange={setSelectedOutline}
                  disabled={!selectedWorldId}
                  searchable
                />
                <Select
                  label="章节正文"
                  data={chapters.map((chapter) => ({
                    value: chapter.id,
                    label: `${chapter.name} (${chapter.id})`,
                  }))}
                  value={selectedChapter}
                  onChange={setSelectedChapter}
                  disabled={!selectedWorldId}
                  searchable
                  clearable
                />
                <Text size="xs" c="dimmed">当前世界：{selectedWorldRecord?.summary || selectedWorldRecord?.name || '未选择'}</Text>
                <Text size="xs" c="dimmed">当前大纲：{selectedOutlineRecord?.summary || '无摘要'}</Text>
                <Text size="xs" c="dimmed">当前章节：{selectedChapterRecord?.content?.slice(0, 120) || '未选择'}</Text>
              </Stack>
            </Paper>

            <Paper p="md" radius="sm" withBorder>
              <Stack gap="sm">
                <Title order={4}>严格验证日志</Title>
                <ScrollArea h={360}>
                  <Stack gap="xs">
                    {logs.map((log) => (
                      <Alert key={log.id} color={log.level === 'error' ? 'red' : log.level === 'success' ? 'green' : 'blue'} variant="light">
                        <Text size="sm">{log.message}</Text>
                      </Alert>
                    ))}
                  </Stack>
                </ScrollArea>
              </Stack>
            </Paper>
          </Stack>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 6 }}>
          <Paper h={560} p="xs" radius="sm" withBorder>
            <ReactFlow nodes={dbGraph.nodes} edges={dbGraph.edges} fitView>
              <MiniMap />
              <Controls />
              <Background />
            </ReactFlow>
          </Paper>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 3 }}>
          <Paper p="md" radius="sm" withBorder>
            <Tabs defaultValue="outline">
              <Tabs.List grow>
                <Tabs.Tab value="novel">小说</Tabs.Tab>
                <Tabs.Tab value="outline">大纲</Tabs.Tab>
                <Tabs.Tab value="chapter">章节</Tabs.Tab>
              </Tabs.List>

              <Tabs.Panel value="novel" pt="md">
                <Stack gap="sm">
                  <TextInput label="小说名称" value={outlineForm.name} onChange={(event) => setOutlineForm({ ...outlineForm, name: event.currentTarget.value })} />
                  <Textarea label="小说简介" minRows={3} value={outlineForm.summary} onChange={(event) => setOutlineForm({ ...outlineForm, summary: event.currentTarget.value })} />
                  <Button leftSection={<IconPlayerPlay size={16} />} onClick={() => openWorkflow('novel', 'create')} disabled={!selectedWorldId}>创建小说工作流</Button>
                  <Divider />
                  <Select
                    label="待修改小说"
                    data={novels.map((novel) => ({ value: novel.novel_id, label: `${novel.name} (${novel.novel_id})` }))}
                    value={selectedNovel}
                    onChange={setSelectedNovel}
                    disabled={!selectedWorldId}
                    searchable
                  />
                  <TextInput label="修改后名称" value={outlineForm.updatedName} onChange={(event) => setOutlineForm({ ...outlineForm, updatedName: event.currentTarget.value })} />
                  <Textarea label="修改后简介" minRows={3} value={outlineForm.updatedContent} onChange={(event) => setOutlineForm({ ...outlineForm, updatedContent: event.currentTarget.value })} />
                  <Button leftSection={<IconEdit size={16} />} variant="light" onClick={() => openWorkflow('novel', 'update')} disabled={!selectedWorldId || !selectedNovel}>修改小说工作流</Button>
                </Stack>
              </Tabs.Panel>

              <Tabs.Panel value="outline" pt="md">
                <Stack gap="sm">
                  <TextInput label="新大纲名称" value={outlineForm.name} onChange={(event) => setOutlineForm({ ...outlineForm, name: event.currentTarget.value })} />
                  <Textarea label="新大纲摘要" minRows={3} value={outlineForm.summary} onChange={(event) => setOutlineForm({ ...outlineForm, summary: event.currentTarget.value })} />
                  <Button leftSection={<IconPlayerPlay size={16} />} onClick={() => openWorkflow('outline', 'create')} disabled={!selectedWorldId || !selectedNovel}>创建大纲工作流</Button>
                  <Divider />
                  <TextInput label="修改后标题" value={outlineForm.updatedName} onChange={(event) => setOutlineForm({ ...outlineForm, updatedName: event.currentTarget.value })} />
                  <Textarea label="修改后内容" minRows={4} value={outlineForm.updatedContent} onChange={(event) => setOutlineForm({ ...outlineForm, updatedContent: event.currentTarget.value })} />
                  <Button leftSection={<IconEdit size={16} />} variant="light" onClick={() => openWorkflow('outline', 'update')} disabled={!selectedWorldId || !selectedOutline}>修改大纲工作流</Button>
                  <Button leftSection={<IconTrash size={16} />} color="red" variant="light" onClick={() => setHumanMode('delete-outline')} disabled={!selectedWorldId || !selectedOutline}>删除表单</Button>
                </Stack>
              </Tabs.Panel>

              <Tabs.Panel value="chapter" pt="md">
                <Stack gap="sm">
                  <TextInput label="章节 ID（留空自动生成）" value={chapterForm.id} onChange={(event) => setChapterForm({ ...chapterForm, id: event.currentTarget.value })} />
                  <TextInput label="章节标题" value={chapterForm.title} onChange={(event) => setChapterForm({ ...chapterForm, title: event.currentTarget.value })} />
                  <Textarea label="章节正文" minRows={5} value={chapterForm.content} onChange={(event) => setChapterForm({ ...chapterForm, content: event.currentTarget.value })} />
                  <Button leftSection={<IconPlayerPlay size={16} />} onClick={() => openWorkflow('chapter', 'create')} disabled={!selectedWorldId || !selectedOutline}>创建章节工作流</Button>
                  <Divider />
                  <TextInput label="修改后标题" value={chapterForm.updatedTitle} onChange={(event) => setChapterForm({ ...chapterForm, updatedTitle: event.currentTarget.value })} />
                  <Textarea label="修改后正文" minRows={4} value={chapterForm.updatedContent} onChange={(event) => setChapterForm({ ...chapterForm, updatedContent: event.currentTarget.value })} />
                  <Button leftSection={<IconEdit size={16} />} variant="light" onClick={() => openWorkflow('chapter', 'update')} disabled={!selectedWorldId || !selectedChapter}>修改章节工作流</Button>
                  <Button leftSection={<IconTrash size={16} />} color="red" variant="light" onClick={() => setHumanMode('delete-chapter')} disabled={!selectedWorldId || !selectedChapter}>删除表单</Button>
                </Stack>
              </Tabs.Panel>

            </Tabs>
          </Paper>
        </Grid.Col>
      </Grid>

      <Paper p="md" radius="sm" withBorder>
        <Title order={4}>最近一次真实响应</Title>
        <JsonInput value={JSON.stringify(lastResponse || {}, null, 2)} readOnly autosize minRows={8} />
      </Paper>

      <Modal opened={humanMode !== null} onClose={() => setHumanMode(null)} title="人工介入表单" size="lg">
        <Stack gap="sm">
          <Alert color="red" icon={<IconAlertTriangle size={18} />}>删除必须填写原因；提交后会立即查询验证是否真的删除。</Alert>
          <Textarea
            label="删除原因"
            required
            minRows={4}
            value={humanForm.deleteReason}
            onChange={(event) => setHumanForm({ ...humanForm, deleteReason: event.currentTarget.value })}
          />
          <Button color="red" onClick={humanMode === 'delete-outline' ? deleteOutline : deleteChapter}>确认删除并查询验证</Button>
        </Stack>
      </Modal>

      <Modal opened={importModalOpen} onClose={() => setImportModalOpen(false)} title="导入世界观层级" size="lg">
        <Stack gap="sm">
          <Alert color="blue">
            支持 JSON、Markdown、XML、OPML。导入会保留标题/节点的上下级路径，并写入当前世界下选定的世界观。
          </Alert>
          <TextInput
            label="所属世界"
            value={selectedWorldRecord ? `${selectedWorldRecord.name} (${selectedWorldRecord.world_id})` : selectedWorldId}
            readOnly
          />
          <Select
            required
            label="目标世界观"
            data={toWorldviewOptions(worldviews)}
            value={importWorldviewId || null}
            onChange={(value) => setImportWorldviewId(value || '')}
            searchable
          />
          <FileInput
            required
            label="导入文件"
            placeholder="选择 .json / .md / .opml"
            accept=".json,.md,.markdown,.xml,.opml"
            value={importFile}
            onChange={setImportFile}
          />
          <Alert color="yellow">
            导入完成后会立即通过真实查询接口验证导入条目是否存在；验证失败会显示错误，不会伪造成功。
          </Alert>
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setImportModalOpen(false)}>取消</Button>
            <Button leftSection={<IconFileImport size={16} />} loading={importing} onClick={importWorldviewHierarchy}>导入并查询验证</Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
};
