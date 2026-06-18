import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  Alert,
  Badge,
  Box,
  Button,
  Group,
  Loader,
  Modal,
  Pagination,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconCircleCheck,
  IconEdit,
  IconFileText,
  IconEye,
  IconPlus,
  IconRefresh,
  IconTrash,
} from '@tabler/icons-react';
import { api, getApiErrorMessage, isApiNotFoundError } from '../api/client';
import { getWorldviewDisplayName } from '../utils/worldview';

type Novel = {
  novel_id: string;
  world_id: string;
  name: string;
};

type Outline = {
  outline_id: string;
  novel_id?: string;
  world_id?: string;
  worldview_id?: string;
  name?: string;
  title?: string;
  summary?: string;
};

type Worldview = {
  worldview_id: string;
  name?: string;
  title?: string;
};

type Chapter = {
  id: string;
  scene_id?: string;
  prose_id?: string;
  chapter_outline_id?: string;
  type?: string;
  name?: string;
  title?: string;
  content?: string;
  outline_id?: string;
  novel_id?: string;
  worldview_id?: string;
  world_id?: string;
  updated_at?: string;
  created_at?: string;
};

const pageSize = 20;

export const NovelChapterManagement: React.FC = () => {
  const { novelId = '' } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const [novel, setNovel] = useState<Novel | null>(null);
  const [worldviews, setWorldviews] = useState<Worldview[]>([]);
  const [outlines, setOutlines] = useState<Outline[]>([]);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [selectedOutlineId, setSelectedOutlineId] = useState(searchParams.get('outline_id') || '');
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [missingNovel, setMissingNovel] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [createChapterOpened, setCreateChapterOpened] = useState(false);
  const [createChapterName, setCreateChapterName] = useState('');
  const [createChapterContent, setCreateChapterContent] = useState('');
  const [createChapterOutlineId, setCreateChapterOutlineId] = useState('');
  const [createChapterWorldviewId, setCreateChapterWorldviewId] = useState('');
  const [editChapterOpened, setEditChapterOpened] = useState(false);
  const [editingChapter, setEditingChapter] = useState<Chapter | null>(null);
  const [editChapterName, setEditChapterName] = useState('');
  const [editChapterContent, setEditChapterContent] = useState('');
  const [editChapterOutlineId, setEditChapterOutlineId] = useState('');
  const [editChapterWorldviewId, setEditChapterWorldviewId] = useState('');
  const [deleteChapter, setDeleteChapter] = useState<Chapter | null>(null);
  const [viewChapter, setViewChapter] = useState<Chapter | null>(null);

  const chapterRecordId = useCallback(
    (chapter: Chapter) => chapter.id || chapter.scene_id || chapter.prose_id || '',
    [],
  );

  const selectedOutline = useMemo(
    () => outlines.find((outline) => outline.outline_id === selectedOutlineId),
    [outlines, selectedOutlineId],
  );
  const hasNextPage = chapters.length === pageSize;
  const paginationTotal = Math.max(page + (hasNextPage ? 1 : 0), 1);

  const loadData = useCallback(async () => {
    if (!novelId) return;
    setLoading(true);
    setError(null);
    setMissingNovel(false);
    try {
      const novelRes = await api.getNovel({ novel_id: novelId });
      const nextNovel = novelRes.data.novel as Novel;
      const [outlineRes, worldviewRes] = await Promise.all([
        api.listOutlines({ novel_id: novelId, page: 1, page_size: 100 }),
        api.listWorldviews({ world_id: nextNovel.world_id, page: 1, page_size: 100 }),
      ]);
      const nextOutlines = ((outlineRes.data as Outline[]) || []).filter((outline) => outline.novel_id === novelId);
      const loreRes = await api.listLore({
        world_id: nextNovel.world_id,
        novel_id: novelId,
        outline_id: selectedOutlineId || undefined,
        chapter_outline_mode: 'root',
        type: 'prose',
        page,
        page_size: pageSize,
      });
      const nextChapters = ((loreRes.data as Chapter[]) || []).filter(
        (chapter) => chapter.type === 'prose' && chapter.novel_id === novelId && !chapter.chapter_outline_id,
      );
      setNovel(nextNovel);
      setWorldviews((worldviewRes.data as Worldview[]) || []);
      setOutlines(nextOutlines);
      setChapters(nextChapters);
    } catch (err: unknown) {
      const nextMissingNovel = isApiNotFoundError(err, 'Novel not found');
      setMissingNovel(nextMissingNovel);
      setError(
        nextMissingNovel
          ? `小说 ${novelId} 不存在，可能已被删除。请返回小说列表重新选择。`
          : getApiErrorMessage(err, '加载分卷章节大纲失败。'),
      );
      setNovel(null);
      setWorldviews([]);
      setOutlines([]);
      setChapters([]);
    } finally {
      setLoading(false);
    }
  }, [novelId, page, selectedOutlineId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    const nextOutlineId = searchParams.get('outline_id') || '';
    setSelectedOutlineId(nextOutlineId);
    setPage(1);
  }, [searchParams]);

  const openCreateWorkflow = () => {
    const targetOutline = selectedOutline || outlines[0];
    if (!novel || !targetOutline) return;
    const params = new URLSearchParams({
      action: 'create',
      world_id: novel.world_id,
      novel_id: novel.novel_id,
      outline_id: targetOutline.outline_id,
    });
    if (targetOutline.worldview_id) params.set('worldview_id', targetOutline.worldview_id);
    navigate(`/workflow/chapter?${params.toString()}`);
  };

  const openCreateChapterModal = () => {
    const targetOutline = selectedOutline || outlines[0];
    if (!novel || !targetOutline) return;
    setCreateChapterName('');
    setCreateChapterContent('');
    setCreateChapterOutlineId(targetOutline.outline_id);
    setCreateChapterWorldviewId(targetOutline.worldview_id || worldviews[0]?.worldview_id || '');
    setCreateChapterOpened(true);
  };

  const openUpdateWorkflow = (chapter: Chapter) => {
    if (!novel) return;
    const chapterId = chapter.id || chapter.scene_id || chapter.prose_id || '';
    const outlineId = chapter.outline_id || selectedOutlineId;
    const outline = outlines.find((item) => item.outline_id === outlineId);
    const params = new URLSearchParams({
      action: 'update',
      world_id: chapter.world_id || novel.world_id,
      novel_id: novel.novel_id,
      outline_id: outlineId,
      id: chapterId,
      name: chapter.name || chapter.title || '',
      content: chapter.content || '',
    });
    if (chapter.worldview_id || outline?.worldview_id) {
      params.set('worldview_id', chapter.worldview_id || outline?.worldview_id || '');
    }
    navigate(`/workflow/chapter?${params.toString()}`);
  };

  const openChapterContentManagement = (chapter: Chapter) => {
    if (!novel) return;
    const chapterId = chapterRecordId(chapter);
    const params = new URLSearchParams({
      outline_id: chapter.outline_id || selectedOutlineId || '',
      chapter_outline_id: chapterId,
    });
    navigate(`/novels/${encodeURIComponent(novel.novel_id)}/chapter-contents?${params.toString()}`);
  };

  const openEditChapterModal = (chapter: Chapter) => {
    setEditingChapter(chapter);
    setEditChapterName(chapter.name || chapter.title || '');
    setEditChapterContent(chapter.content || '');
    setEditChapterOutlineId(chapter.outline_id || selectedOutlineId || outlines[0]?.outline_id || '');
    setEditChapterWorldviewId(chapter.worldview_id || selectedOutline?.worldview_id || worldviews[0]?.worldview_id || '');
    setEditChapterOpened(true);
  };

  const openCheckWorkflow = (chapter?: Chapter) => {
    if (!novel) return;
    const targetOutline = chapter?.outline_id || selectedOutlineId || outlines[0]?.outline_id || '';
    if (!targetOutline) return;
    const outline = outlines.find((item) => item.outline_id === targetOutline);
    const chapterId = chapter ? (chapter.id || chapter.scene_id || chapter.prose_id || '') : '';
    const params = new URLSearchParams({
      action: 'check',
      world_id: chapter?.world_id || novel.world_id,
      novel_id: chapter?.novel_id || novel.novel_id,
      outline_id: targetOutline,
    });
    if (chapterId) {
      params.set('id', chapterId);
    }
    if (chapter?.name || chapter?.title) {
      params.set('name', chapter?.name || chapter?.title || '');
    }
    if (chapter?.worldview_id || outline?.worldview_id) {
      params.set('worldview_id', chapter?.worldview_id || outline?.worldview_id || '');
    }
    navigate(`/workflow/chapter?${params.toString()}`);
  };

  const confirmDelete = async () => {
    if (!deleteChapter || !novel) return;
    const chapterId = deleteChapter.id || deleteChapter.scene_id || deleteChapter.prose_id || '';
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteArchiveItem({
        id: chapterId,
        type: 'prose',
        world_id: deleteChapter.world_id || novel.world_id,
        novel_id: novel.novel_id,
        outline_id: deleteChapter.outline_id,
      });
      const verifyRes = await api.listLore({
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        outline_id: deleteChapter.outline_id,
        chapter_outline_mode: 'root',
        type: 'prose',
        page: 1,
        page_size: 100,
      });
      const remaining = ((verifyRes.data as Chapter[]) || []).filter((chapter) => chapterRecordId(chapter) === chapterId);
      if (remaining.length > 0) {
        throw new Error(`删除后仍能查询到分卷章节大纲: ${chapterId}`);
      }
      setSuccess('分卷章节大纲已删除，并完成真实接口回查。');
      setDeleteChapter(null);
      await loadData();
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '删除分卷章节大纲失败。'));
    } finally {
      setSaving(false);
    }
  };

  const confirmCreateChapter = async () => {
    if (!novel) return;
    const nextName = createChapterName.trim();
    const targetOutlineId = createChapterOutlineId || selectedOutlineId || outlines[0]?.outline_id || '';
    if (!targetOutlineId) {
      setError('必须先选择分卷大纲，才能新增分卷章节大纲。');
      return;
    }
    if (!nextName) {
      setError('章节标题不能为空。');
      return;
    }
    const chapterId = `prose_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateArchiveItem({
        id: chapterId,
        type: 'prose',
        name: nextName,
        content: createChapterContent,
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        outline_id: targetOutlineId,
        worldview_id: createChapterWorldviewId || undefined,
      });
      const verifyRes = await api.listLore({
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        chapter_outline_mode: 'root',
        type: 'prose',
        page: 1,
        page_size: 200,
      });
      const created = ((verifyRes.data as Chapter[]) || []).find((chapter) => chapterRecordId(chapter) === chapterId);
      if (!created) {
        throw new Error(`新增后未查询到分卷章节大纲: ${chapterId}`);
      }
      if ((created.name || created.title || '') !== nextName) {
        throw new Error(`章节标题未写入，实际为: ${created.name || created.title || ''}`);
      }
      if ((created.content || '') !== createChapterContent) {
        throw new Error('章节内容未写入为提交内容。');
      }
      if ((created.outline_id || '') !== targetOutlineId) {
        throw new Error(`章节所属分卷未写入，实际为: ${created.outline_id || ''}`);
      }
      setSuccess(`分卷章节大纲已直接创建，并完成真实接口回查：${chapterId}`);
      setCreateChapterOpened(false);
      setCreateChapterName('');
      setCreateChapterContent('');
      if (selectedOutlineId !== targetOutlineId) {
        navigate(`/novels/${encodeURIComponent(novel.novel_id)}/chapters?outline_id=${encodeURIComponent(targetOutlineId)}`);
      } else {
        await loadData();
      }
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '直接新增分卷章节大纲失败。'));
    } finally {
      setSaving(false);
    }
  };

  const confirmUpdateChapter = async () => {
    if (!novel || !editingChapter) return;
    const chapterId = editingChapter.id || editingChapter.scene_id || editingChapter.prose_id || '';
    const nextName = editChapterName.trim();
    const targetOutlineId = editChapterOutlineId || editingChapter.outline_id || selectedOutlineId || '';
    if (!chapterId) {
      setError('缺少章节 ID，无法直接修改。');
      return;
    }
    if (!targetOutlineId) {
      setError('章节必须归属于分卷大纲。');
      return;
    }
    if (!nextName) {
      setError('章节标题不能为空。');
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateArchiveItem({
        id: chapterId,
        type: 'prose',
        name: nextName,
        content: editChapterContent,
        world_id: editingChapter.world_id || novel.world_id,
        novel_id: novel.novel_id,
        outline_id: targetOutlineId,
        worldview_id: editChapterWorldviewId || undefined,
      });
      const verifyRes = await api.listLore({
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        chapter_outline_mode: 'root',
        type: 'prose',
        page: 1,
        page_size: 200,
      });
      const updated = ((verifyRes.data as Chapter[]) || []).find((chapter) => chapterRecordId(chapter) === chapterId);
      if (!updated) {
        throw new Error(`修改后未查询到章节内容: ${chapterId}`);
      }
      if ((updated.name || updated.title || '') !== nextName) {
        throw new Error(`章节标题未更新，实际为: ${updated.name || updated.title || ''}`);
      }
      if ((updated.content || '') !== editChapterContent) {
        throw new Error('章节内容未更新为提交内容。');
      }
      if ((updated.outline_id || '') !== targetOutlineId) {
        throw new Error(`章节所属分卷未更新，实际为: ${updated.outline_id || ''}`);
      }
      if ((editChapterWorldviewId || '') !== (updated.worldview_id || '')) {
        throw new Error(`章节世界观未更新，实际为: ${updated.worldview_id || ''}`);
      }
      setSuccess(`章节内容已直接修改，并完成真实接口回查：${chapterId}`);
      setEditChapterOpened(false);
      setEditingChapter(null);
      if (selectedOutlineId !== targetOutlineId) {
        navigate(`/novels/${encodeURIComponent(novel.novel_id)}/chapters?outline_id=${encodeURIComponent(targetOutlineId)}`);
      } else {
        await loadData();
      }
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '直接修改章节内容失败。'));
    } finally {
      setSaving(false);
    }
  };

  const title = useMemo(() => novel?.name || novelId, [novel?.name, novelId]);
  const outlineNameById = useMemo(
    () => Object.fromEntries(outlines.map((outline) => [outline.outline_id, outline.name || outline.title || outline.outline_id])),
    [outlines],
  );
  const worldviewNameById = useMemo(
    () => Object.fromEntries(worldviews.map((worldview) => [worldview.worldview_id, getWorldviewDisplayName(worldview)])),
    [worldviews],
  );

  return (
    <Stack gap="md" style={{ height: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Group gap="xs">
            <IconFileText size={24} />
            <Title order={2}>分卷章节大纲管理</Title>
            <Badge variant="light">{novelId}</Badge>
          </Group>
          <Text size="sm" c="dimmed">当前小说：{title}。本页统一管理分卷章节大纲；当前 `prose` 记录保存章节标题与对应章节大纲正文。</Text>
        </Box>
        <Group gap="xs">
          <Button variant="light" leftSection={<IconArrowLeft size={16} />} onClick={() => navigate(`/novels/${encodeURIComponent(novelId)}/outlines`)}>返回分卷大纲页</Button>
          <Button variant="light" leftSection={<IconRefresh size={16} />} loading={loading} onClick={loadData}>刷新</Button>
          <Button variant="light" leftSection={<IconCircleCheck size={16} />} onClick={() => openCheckWorkflow()} disabled={!novel || outlines.length === 0}>检查直接内容</Button>
          <Button variant="light" leftSection={<IconPlus size={16} />} onClick={openCreateWorkflow} disabled={!novel || outlines.length === 0}>工作流新增分卷章节大纲</Button>
          <Button leftSection={<IconPlus size={16} />} onClick={openCreateChapterModal} disabled={!novel || outlines.length === 0}>直接新增分卷章节大纲</Button>
        </Group>
      </Group>

      {error && (
        <Alert color={missingNovel ? 'yellow' : 'red'} title={missingNovel ? '目标小说不存在或已删除' : '真实接口请求失败'}>
          {error}
        </Alert>
      )}
      {success && <Alert color="green">{success}</Alert>}

      <Paper p="md" withBorder>
        <Group justify="space-between" align="flex-end">
          <Select
            label="分卷大纲筛选"
            data={outlines.map((outline) => ({ value: outline.outline_id, label: `${outline.name || outline.title || outline.outline_id} (${outline.outline_id})` }))}
            value={selectedOutlineId || null}
            onChange={(value) => { setSelectedOutlineId(value || ''); setPage(1); }}
            clearable
            searchable
            placeholder="全部分卷"
            style={{ minWidth: 420 }}
          />
          <Group gap="xs">
            {loading && <Loader size="sm" />}
            <Badge variant="light">第 {page} 页</Badge>
            <Badge variant="light">本页 {chapters.length} 条</Badge>
          </Group>
        </Group>
        {!selectedOutlineId && <Text size="xs" c="dimmed" mt="xs">未选择分卷时表格显示全部章节记录；直接新增分卷章节大纲时默认使用当前小说下第一条分卷。若当前还没有分卷，请先到分卷大纲页创建父级分卷。</Text>}
      </Paper>

      <Paper p="sm" withBorder style={{ overflow: 'hidden', flex: 1 }}>
        <Group justify="flex-end" mb="sm">
          <Pagination value={page} onChange={setPage} total={paginationTotal} />
        </Group>
        <ScrollArea h="calc(100vh - 310px)">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>章节大纲 ID</Table.Th>
                <Table.Th>标题</Table.Th>
                <Table.Th>所属分卷</Table.Th>
                <Table.Th>正文摘要</Table.Th>
                <Table.Th>更新时间</Table.Th>
                <Table.Th>操作</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {chapters.map((chapter) => {
                const chapterId = chapterRecordId(chapter);
                const outlineName = chapter.outline_id ? (outlineNameById[chapter.outline_id] || chapter.outline_id) : '';
                return (
                  <Table.Tr key={chapterId}>
                    <Table.Td><Text size="xs" truncate maw={180}>{chapterId}</Text></Table.Td>
                    <Table.Td><Text fw={700}>{chapter.name || chapter.title || ''}</Text></Table.Td>
                    <Table.Td><Text size="xs" truncate maw={170}>{outlineName}</Text></Table.Td>
                    <Table.Td><Text size="sm" truncate maw={520}>{chapter.content || ''}</Text></Table.Td>
                    <Table.Td><Text size="xs" c="dimmed">{chapter.updated_at || chapter.created_at || ''}</Text></Table.Td>
                    <Table.Td>
                      <Group gap="xs" wrap="nowrap">
                        <Button size="xs" variant="light" leftSection={<IconEye size={14} />} onClick={() => setViewChapter(chapter)}>查看</Button>
                        <Button size="xs" variant="light" leftSection={<IconFileText size={14} />} onClick={() => openChapterContentManagement(chapter)}>查看章节内容</Button>
                        <Button size="xs" variant="light" leftSection={<IconCircleCheck size={14} />} onClick={() => openCheckWorkflow(chapter)}>检查内容</Button>
                        <Button size="xs" variant="light" leftSection={<IconEdit size={14} />} onClick={() => openEditChapterModal(chapter)}>直接修改</Button>
                        <Button size="xs" variant="light" onClick={() => openUpdateWorkflow(chapter)}>工作流修改分卷章节大纲</Button>
                        <Button size="xs" color="red" variant="light" leftSection={<IconTrash size={14} />} onClick={() => setDeleteChapter(chapter)}>删除</Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
              {chapters.length === 0 && (
                <Table.Tr>
                  <Table.Td colSpan={6}>
                    <Text ta="center" c="dimmed" py="xl">当前筛选下暂无分卷章节大纲。</Text>
                  </Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Paper>

      <Modal opened={Boolean(viewChapter)} onClose={() => setViewChapter(null)} title="查看分卷章节大纲" size="lg">
        <Stack gap="sm">
          <Text fw={700}>{viewChapter?.name || viewChapter?.title}</Text>
          <Text size="xs" c="dimmed">章节大纲 ID：{viewChapter?.id || viewChapter?.scene_id || viewChapter?.prose_id}</Text>
          <Text size="xs" c="dimmed">所属分卷：{viewChapter?.outline_id ? (outlineNameById[viewChapter.outline_id] || viewChapter.outline_id) : ''}</Text>
          <Paper withBorder p="sm">
            <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{viewChapter?.content || '暂无内容。'}</Text>
          </Paper>
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setViewChapter(null)}>关闭</Button>
            <Button variant="light" onClick={() => { if (viewChapter) { setViewChapter(null); openEditChapterModal(viewChapter); } }}>直接修改</Button>
            <Button onClick={() => { if (viewChapter) { setViewChapter(null); openUpdateWorkflow(viewChapter); } }}>工作流修改分卷章节大纲</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={Boolean(deleteChapter)} onClose={() => setDeleteChapter(null)} title="删除分卷章节大纲" size="md">
        <Stack gap="sm">
          <Alert color="red">删除会移除该分卷章节大纲记录。</Alert>
          <Text fw={700}>{deleteChapter?.name || deleteChapter?.title}</Text>
          <Text size="xs" c="dimmed">{deleteChapter?.id || deleteChapter?.scene_id || deleteChapter?.prose_id}</Text>
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setDeleteChapter(null)}>取消</Button>
            <Button color="red" loading={saving} onClick={confirmDelete}>确认删除</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={createChapterOpened} onClose={() => setCreateChapterOpened(false)} title="直接新增分卷章节大纲" size="lg">
        <Stack gap="sm">
          <Alert color="blue">这个入口直接写入 `prose`，不走工作流；适合直接录入分卷章节大纲全文。</Alert>
          <TextInput
            label="章节标题"
            placeholder="例如：第一章：坠落"
            value={createChapterName}
            onChange={(event) => setCreateChapterName(event.currentTarget.value)}
          />
          <Select
            label="所属分卷"
            data={outlines.map((outline) => ({ value: outline.outline_id, label: `${outline.name || outline.title || outline.outline_id} (${outline.outline_id})` }))}
            value={createChapterOutlineId || null}
            onChange={(value) => setCreateChapterOutlineId(value || '')}
            allowDeselect={false}
            searchable
          />
          <Select
            label="世界观"
            data={worldviews.map((worldview) => ({ value: worldview.worldview_id, label: worldviewNameById[worldview.worldview_id] || worldview.worldview_id }))}
            value={createChapterWorldviewId || null}
            onChange={(value) => setCreateChapterWorldviewId(value || '')}
            clearable
            searchable
          />
          <Textarea
            label="分卷章节大纲正文"
            minRows={12}
            autosize
            value={createChapterContent}
            onChange={(event) => setCreateChapterContent(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setCreateChapterOpened(false)}>取消</Button>
            <Button loading={saving} onClick={confirmCreateChapter}>直接创建</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={editChapterOpened} onClose={() => setEditChapterOpened(false)} title="直接修改分卷章节大纲" size="lg">
        <Stack gap="sm">
          <Alert color="blue">这个入口直接写入 `prose`，不走工作流；保存后会立刻回查数据库结果。</Alert>
          <Text size="xs" c="dimmed">章节 ID：{editingChapter?.id || editingChapter?.scene_id || editingChapter?.prose_id || ''}</Text>
          <TextInput
            label="章节标题"
            placeholder="例如：第一章：坠落"
            value={editChapterName}
            onChange={(event) => setEditChapterName(event.currentTarget.value)}
          />
          <Select
            label="所属分卷"
            data={outlines.map((outline) => ({ value: outline.outline_id, label: `${outline.name || outline.title || outline.outline_id} (${outline.outline_id})` }))}
            value={editChapterOutlineId || null}
            onChange={(value) => setEditChapterOutlineId(value || '')}
            allowDeselect={false}
            searchable
          />
          <Select
            label="世界观"
            data={worldviews.map((worldview) => ({ value: worldview.worldview_id, label: worldviewNameById[worldview.worldview_id] || worldview.worldview_id }))}
            value={editChapterWorldviewId || null}
            onChange={(value) => setEditChapterWorldviewId(value || '')}
            clearable
            searchable
          />
          <Textarea
            label="分卷章节大纲正文"
            minRows={12}
            autosize
            value={editChapterContent}
            onChange={(event) => setEditChapterContent(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setEditChapterOpened(false)}>取消</Button>
            <Button loading={saving} onClick={confirmUpdateChapter}>直接保存</Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
};
