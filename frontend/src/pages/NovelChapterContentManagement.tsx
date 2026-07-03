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
  IconEye,
  IconFileText,
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

type ChapterRecord = {
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

export const NovelChapterContentManagement: React.FC = () => {
  const { novelId = '' } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const [novel, setNovel] = useState<Novel | null>(null);
  const [worldviews, setWorldviews] = useState<Worldview[]>([]);
  const [outlines, setOutlines] = useState<Outline[]>([]);
  const [chapterOutlines, setChapterOutlines] = useState<ChapterRecord[]>([]);
  const [chapterContents, setChapterContents] = useState<ChapterRecord[]>([]);
  const [selectedOutlineId, setSelectedOutlineId] = useState(searchParams.get('outline_id') || '');
  const [selectedChapterOutlineId, setSelectedChapterOutlineId] = useState(searchParams.get('chapter_outline_id') || '');
  const [selectedContentId, setSelectedContentId] = useState('');
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [missingNovel, setMissingNovel] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [createOpened, setCreateOpened] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createContent, setCreateContent] = useState('');
  const [createWorldviewId, setCreateWorldviewId] = useState('');
  const [editOpened, setEditOpened] = useState(false);
  const [editName, setEditName] = useState('');
  const [editContent, setEditContent] = useState('');
  const [editWorldviewId, setEditWorldviewId] = useState('');
  const [viewContent, setViewContent] = useState<ChapterRecord | null>(null);
  const [deleteContent, setDeleteContent] = useState<ChapterRecord | null>(null);

  const recordId = useCallback(
    (record: ChapterRecord | null | undefined) => record?.id || record?.scene_id || record?.prose_id || '',
    [],
  );

  const selectedChapterOutline = useMemo(
    () => chapterOutlines.find((item) => recordId(item) === selectedChapterOutlineId) || null,
    [chapterOutlines, recordId, selectedChapterOutlineId],
  );
  const selectedContent = useMemo(
    () => chapterContents.find((item) => recordId(item) === selectedContentId) || null,
    [chapterContents, recordId, selectedContentId],
  );
  const hasNextPage = chapterContents.length === pageSize;
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
      const resolvedOutlineId = selectedOutlineId || nextOutlines[0]?.outline_id || '';
      const chapterOutlineRes = resolvedOutlineId
        ? await api.listLore({
          world_id: nextNovel.world_id,
          novel_id: novelId,
          outline_id: resolvedOutlineId,
          chapter_outline_mode: 'root',
          type: 'prose',
          page: 1,
          page_size: 200,
        })
        : { data: [] };
      const nextChapterOutlines = ((chapterOutlineRes.data as ChapterRecord[]) || []).filter(
        (item) => item.type === 'prose' && item.novel_id === novelId && !item.chapter_outline_id,
      );
      const resolvedChapterOutlineId = selectedChapterOutlineId || recordId(nextChapterOutlines[0]);
      const chapterContentRes = resolvedOutlineId && resolvedChapterOutlineId
        ? await api.listLore({
          world_id: nextNovel.world_id,
          novel_id: novelId,
          outline_id: resolvedOutlineId,
          chapter_outline_id: resolvedChapterOutlineId,
          type: 'prose',
          page,
          page_size: pageSize,
        })
        : { data: [] };
      const nextChapterContents = ((chapterContentRes.data as ChapterRecord[]) || []).filter(
        (item) => item.type === 'prose' && item.novel_id === novelId && item.chapter_outline_id === resolvedChapterOutlineId,
      );

      setNovel(nextNovel);
      setWorldviews((worldviewRes.data as Worldview[]) || []);
      setOutlines(nextOutlines);
      setChapterOutlines(nextChapterOutlines);
      setChapterContents(nextChapterContents);
      if (resolvedOutlineId !== selectedOutlineId) {
        setSelectedOutlineId(resolvedOutlineId);
      }
      if (resolvedChapterOutlineId !== selectedChapterOutlineId) {
        setSelectedChapterOutlineId(resolvedChapterOutlineId);
      }
    } catch (err: unknown) {
      const nextMissingNovel = isApiNotFoundError(err, 'Novel not found');
      setMissingNovel(nextMissingNovel);
      setError(
        nextMissingNovel
          ? `小说 ${novelId} 不存在，可能已被删除。请返回小说列表重新选择。`
          : getApiErrorMessage(err, '加载章节内容失败。'),
      );
      setNovel(null);
      setWorldviews([]);
      setOutlines([]);
      setChapterOutlines([]);
      setChapterContents([]);
    } finally {
      setLoading(false);
    }
  }, [novelId, page, recordId, selectedChapterOutlineId, selectedOutlineId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    const nextOutlineId = searchParams.get('outline_id') || '';
    const nextChapterOutlineId = searchParams.get('chapter_outline_id') || '';
    setSelectedOutlineId(nextOutlineId);
    setSelectedChapterOutlineId(nextChapterOutlineId);
    setSelectedContentId('');
    setPage(1);
  }, [searchParams]);

  useEffect(() => {
    if (!chapterContents.length) {
      setSelectedContentId('');
      return;
    }
    if (!chapterContents.some((item) => recordId(item) === selectedContentId)) {
      setSelectedContentId(recordId(chapterContents[0]));
    }
  }, [chapterContents, recordId, selectedContentId]);

  const updatePageQuery = (nextOutlineId: string, nextChapterId: string) => {
    const params = new URLSearchParams();
    if (nextOutlineId) params.set('outline_id', nextOutlineId);
    if (nextChapterId) params.set('chapter_outline_id', nextChapterId);
    navigate(`/novels/${encodeURIComponent(novelId)}/chapter-contents${params.toString() ? `?${params.toString()}` : ''}`);
  };

  const openCreateWorkflow = () => {
    if (!novel || !selectedOutlineId || !selectedChapterOutlineId) return;
    const params = new URLSearchParams({
      action: 'create',
      world_id: novel.world_id,
      novel_id: novel.novel_id,
      outline_id: selectedOutlineId,
      chapter_outline_id: selectedChapterOutlineId,
    });
    if (selectedChapterOutline?.worldview_id) {
      params.set('worldview_id', selectedChapterOutline.worldview_id);
    }
    navigate(`/workflow/chapter?${params.toString()}`);
  };

  const openCheckWorkflow = (contentRecord: ChapterRecord | null = selectedContent) => {
    if (!novel || !selectedOutlineId || !selectedChapterOutlineId || !selectedChapterOutline || !contentRecord) return;
    const params = new URLSearchParams({
      action: 'check',
      world_id: contentRecord.world_id || novel.world_id,
      novel_id: novel.novel_id,
      outline_id: selectedOutlineId,
      chapter_outline_id: selectedChapterOutlineId,
      chapter_outline: selectedChapterOutline.content || '',
      name: contentRecord.name || contentRecord.title || '',
      content: contentRecord.content || '',
    });
    if (contentRecord.worldview_id || selectedChapterOutline.worldview_id) {
      params.set('worldview_id', contentRecord.worldview_id || selectedChapterOutline.worldview_id || '');
    }
    navigate(`/workflow/chapter?${params.toString()}`);
  };

  const openContentSummaryWorkflow = (contentRecord: ChapterRecord | null = selectedContent) => {
    if (!novel || !selectedOutlineId || !selectedChapterOutlineId || !contentRecord) return;
    const params = new URLSearchParams({
      type: 'chapter_intro_summary_create',
      action: 'create',
      target_id: recordId(contentRecord),
      world_id: contentRecord.world_id || novel.world_id,
      novel_id: novel.novel_id,
      outline_id: selectedOutlineId,
      chapter_outline_id: selectedChapterOutlineId,
      name: contentRecord.name || contentRecord.title || '',
      content: contentRecord.content || '',
      message: '生成章节简介与总结',
    });
    if (contentRecord.worldview_id || selectedChapterOutline?.worldview_id) {
      params.set('worldview_id', contentRecord.worldview_id || selectedChapterOutline?.worldview_id || '');
    }
    navigate(`/workflow?${params.toString()}`);
  };

  const openCreateModal = () => {
    if (!selectedChapterOutline) return;
    setCreateName('');
    setCreateContent('');
    setCreateWorldviewId(selectedChapterOutline.worldview_id || worldviews[0]?.worldview_id || '');
    setCreateOpened(true);
  };

  const openUpdateWorkflow = (contentRecord: ChapterRecord | null = selectedContent) => {
    if (!novel || !contentRecord || !selectedOutlineId || !selectedChapterOutlineId) return;
    const params = new URLSearchParams({
      action: 'update',
      world_id: contentRecord.world_id || novel.world_id,
      novel_id: novel.novel_id,
      outline_id: selectedOutlineId,
      chapter_outline_id: selectedChapterOutlineId,
      id: recordId(contentRecord),
      name: contentRecord.name || contentRecord.title || '',
      content: contentRecord.content || '',
    });
    if (contentRecord.worldview_id || selectedChapterOutline?.worldview_id) {
      params.set('worldview_id', contentRecord.worldview_id || selectedChapterOutline?.worldview_id || '');
    }
    navigate(`/workflow/chapter?${params.toString()}`);
  };

  const openEditModal = (contentRecord: ChapterRecord | null = selectedContent) => {
    if (!contentRecord) return;
    setSelectedContentId(recordId(contentRecord));
    setEditName(contentRecord.name || contentRecord.title || '');
    setEditContent(contentRecord.content || '');
    setEditWorldviewId(contentRecord.worldview_id || selectedChapterOutline?.worldview_id || worldviews[0]?.worldview_id || '');
    setEditOpened(true);
  };

  const confirmCreate = async () => {
    if (!novel || !selectedOutlineId || !selectedChapterOutlineId) return;
    const nextName = createName.trim();
    if (!nextName) {
      setError('章节内容标题不能为空。');
      return;
    }
    const contentId = `chapter_content_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateArchiveItem({
        id: contentId,
        type: 'prose',
        name: nextName,
        content: createContent,
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        outline_id: selectedOutlineId,
        chapter_outline_id: selectedChapterOutlineId,
        worldview_id: createWorldviewId || undefined,
      });
      const verifyRes = await api.listLore({
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        outline_id: selectedOutlineId,
        chapter_outline_id: selectedChapterOutlineId,
        type: 'prose',
        page: 1,
        page_size: 200,
      });
      const created = ((verifyRes.data as ChapterRecord[]) || []).find((item) => recordId(item) === contentId);
      if (!created) {
        throw new Error(`新增后未查询到章节内容: ${contentId}`);
      }
      if ((created.name || created.title || '') !== nextName) {
        throw new Error(`章节内容标题未写入，实际为: ${created.name || created.title || ''}`);
      }
      if ((created.content || '') !== createContent) {
        throw new Error('章节内容正文未写入为提交内容。');
      }
      if ((created.chapter_outline_id || '') !== selectedChapterOutlineId) {
        throw new Error(`章节内容父级大纲未写入，实际为: ${created.chapter_outline_id || ''}`);
      }
      setSuccess(`章节内容已直接创建，并完成真实接口回查：${contentId}`);
      setCreateOpened(false);
      setCreateName('');
      setCreateContent('');
      setSelectedContentId(contentId);
      await loadData();
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '直接添加章节内容失败。'));
    } finally {
      setSaving(false);
    }
  };

  const confirmUpdate = async () => {
    if (!novel || !selectedContent || !selectedOutlineId || !selectedChapterOutlineId) return;
    const contentId = recordId(selectedContent);
    const nextName = editName.trim();
    if (!contentId) {
      setError('缺少章节内容 ID，无法直接修改。');
      return;
    }
    if (!nextName) {
      setError('章节内容标题不能为空。');
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateArchiveItem({
        id: contentId,
        type: 'prose',
        name: nextName,
        content: editContent,
        world_id: selectedContent.world_id || novel.world_id,
        novel_id: novel.novel_id,
        outline_id: selectedOutlineId,
        chapter_outline_id: selectedChapterOutlineId,
        worldview_id: editWorldviewId || undefined,
      });
      const verifyRes = await api.listLore({
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        outline_id: selectedOutlineId,
        chapter_outline_id: selectedChapterOutlineId,
        type: 'prose',
        page: 1,
        page_size: 200,
      });
      const updated = ((verifyRes.data as ChapterRecord[]) || []).find((item) => recordId(item) === contentId);
      if (!updated) {
        throw new Error(`修改后未查询到章节内容: ${contentId}`);
      }
      if ((updated.name || updated.title || '') !== nextName) {
        throw new Error(`章节内容标题未更新，实际为: ${updated.name || updated.title || ''}`);
      }
      if ((updated.content || '') !== editContent) {
        throw new Error('章节内容正文未更新为提交内容。');
      }
      if ((updated.chapter_outline_id || '') !== selectedChapterOutlineId) {
        throw new Error(`章节内容父级大纲未更新，实际为: ${updated.chapter_outline_id || ''}`);
      }
      if ((editWorldviewId || '') !== (updated.worldview_id || '')) {
        throw new Error(`章节内容世界观未更新，实际为: ${updated.worldview_id || ''}`);
      }
      setSuccess(`章节内容已直接修改，并完成真实接口回查：${contentId}`);
      setEditOpened(false);
      await loadData();
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '直接修改章节内容失败。'));
    } finally {
      setSaving(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleteContent || !novel || !selectedOutlineId || !selectedChapterOutlineId) return;
    const contentId = recordId(deleteContent);
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteArchiveItem({
        id: contentId,
        type: 'prose',
        world_id: deleteContent.world_id || novel.world_id,
        novel_id: novel.novel_id,
        outline_id: selectedOutlineId,
        chapter_outline_id: selectedChapterOutlineId,
      });
      const verifyRes = await api.listLore({
        world_id: novel.world_id,
        novel_id: novel.novel_id,
        outline_id: selectedOutlineId,
        chapter_outline_id: selectedChapterOutlineId,
        type: 'prose',
        page: 1,
        page_size: 200,
      });
      const remaining = ((verifyRes.data as ChapterRecord[]) || []).filter((item) => recordId(item) === contentId);
      if (remaining.length > 0) {
        throw new Error(`删除后仍能查询到章节内容: ${contentId}`);
      }
      setSuccess(`章节内容已删除，并完成真实接口回查：${contentId}`);
      setDeleteContent(null);
      await loadData();
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '删除章节内容失败。'));
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
            <Title order={2}>章节内容管理</Title>
            <Badge variant="light">{novelId}</Badge>
          </Group>
          <Text size="sm" c="dimmed">
            当前小说：{title}。本页管理具体章节大纲下的章节内容，真实读写 `prose.chapter_outline_id` 子记录。
          </Text>
        </Box>
        <Group gap="xs">
          <Button variant="light" leftSection={<IconArrowLeft size={16} />} onClick={() => navigate(`/novels/${encodeURIComponent(novelId)}/chapters${selectedOutlineId ? `?outline_id=${encodeURIComponent(selectedOutlineId)}` : ''}`)}>
            返回章节大纲页
          </Button>
          <Button variant="light" leftSection={<IconRefresh size={16} />} loading={loading} onClick={loadData}>
            刷新
          </Button>
          <Button leftSection={<IconPlus size={16} />} onClick={openCreateModal} disabled={!selectedChapterOutlineId}>
            直接添加章节内容
          </Button>
          <Button variant="light" leftSection={<IconPlus size={16} />} onClick={openCreateWorkflow} disabled={!selectedChapterOutlineId}>
            工作流添加章节内容
          </Button>
          <Button leftSection={<IconEdit size={16} />} onClick={() => openEditModal()} disabled={!selectedContent}>
            直接修改章节内容
          </Button>
          <Button variant="light" leftSection={<IconEdit size={16} />} onClick={() => openUpdateWorkflow()} disabled={!selectedContent}>
            工作流修改章节内容
          </Button>
          <Button variant="light" leftSection={<IconCircleCheck size={16} />} onClick={() => openCheckWorkflow()} disabled={!selectedContent}>
            检查章节内容
          </Button>
          <Button variant="light" leftSection={<IconFileText size={16} />} onClick={() => openContentSummaryWorkflow()} disabled={!selectedContent}>
            生成章节简介与总结
          </Button>
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
          <Group align="flex-end">
            <Select
              label="所属分卷"
              data={outlines.map((outline) => ({ value: outline.outline_id, label: `${outline.name || outline.title || outline.outline_id} (${outline.outline_id})` }))}
              value={selectedOutlineId || null}
              onChange={(value) => updatePageQuery(value || '', '')}
              searchable
              allowDeselect={false}
              style={{ minWidth: 360 }}
            />
            <Select
              label="具体章节大纲"
              data={chapterOutlines.map((item) => ({ value: recordId(item), label: `${item.name || item.title || recordId(item)} (${recordId(item)})` }))}
              value={selectedChapterOutlineId || null}
              onChange={(value) => updatePageQuery(selectedOutlineId, value || '')}
              searchable
              allowDeselect={false}
              style={{ minWidth: 420 }}
            />
          </Group>
          <Group gap="xs">
            {loading && <Loader size="sm" />}
            <Badge variant="light">第 {page} 页</Badge>
            <Badge variant="light">本页 {chapterContents.length} 条</Badge>
          </Group>
        </Group>
        <Stack gap={4} mt="sm">
          <Text size="xs" c="dimmed">当前分卷：{selectedOutlineId ? (outlineNameById[selectedOutlineId] || selectedOutlineId) : '未选择'}</Text>
          <Text size="xs" c="dimmed">当前章节大纲：{selectedChapterOutline ? `${selectedChapterOutline.name || selectedChapterOutline.title || recordId(selectedChapterOutline)} (${recordId(selectedChapterOutline)})` : '未选择'}</Text>
          <Text size="xs" c="dimmed">当前选中章节内容：{selectedContent ? `${selectedContent.name || selectedContent.title || recordId(selectedContent)} (${recordId(selectedContent)})` : '未选择'}</Text>
        </Stack>
        {selectedChapterOutline?.content && (
          <Paper withBorder p="sm" mt="sm">
            <Text fw={700} size="sm">父级章节大纲正文</Text>
            <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{selectedChapterOutline.content}</Text>
          </Paper>
        )}
      </Paper>

      <Paper p="sm" withBorder style={{ overflow: 'hidden', flex: 1 }}>
        <Group justify="flex-end" mb="sm">
          <Pagination value={page} onChange={setPage} total={paginationTotal} />
        </Group>
        <ScrollArea h="calc(100vh - 360px)">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>章节内容 ID</Table.Th>
                <Table.Th>标题</Table.Th>
                <Table.Th>正文摘要</Table.Th>
                <Table.Th>更新时间</Table.Th>
                <Table.Th>操作</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {chapterContents.map((item) => {
                const contentId = recordId(item);
                const isSelected = contentId === selectedContentId;
                return (
                  <Table.Tr key={contentId}>
                    <Table.Td>
                      <Group gap="xs">
                        <Text size="xs" truncate maw={180}>{contentId}</Text>
                        {isSelected && <Badge size="xs">当前</Badge>}
                      </Group>
                    </Table.Td>
                    <Table.Td><Text fw={700}>{item.name || item.title || ''}</Text></Table.Td>
                    <Table.Td><Text size="sm" truncate maw={620}>{item.content || ''}</Text></Table.Td>
                    <Table.Td><Text size="xs" c="dimmed">{item.updated_at || item.created_at || ''}</Text></Table.Td>
                    <Table.Td>
                      <Group gap="xs" wrap="nowrap">
                        <Button size="xs" variant="light" leftSection={<IconEye size={14} />} onClick={() => { setSelectedContentId(contentId); setViewContent(item); }}>
                          查看
                        </Button>
                        <Button size="xs" variant="light" onClick={() => setSelectedContentId(contentId)}>
                          设为当前
                        </Button>
                        <Button size="xs" color="red" variant="light" leftSection={<IconTrash size={14} />} onClick={() => { setSelectedContentId(contentId); setDeleteContent(item); }}>
                          删除
                        </Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
              {chapterContents.length === 0 && (
                <Table.Tr>
                  <Table.Td colSpan={5}>
                    <Text ta="center" c="dimmed" py="xl">当前章节大纲下暂无章节内容。</Text>
                  </Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Paper>

      <Modal opened={Boolean(viewContent)} onClose={() => setViewContent(null)} title="查看章节内容" size="lg">
        <Stack gap="sm">
          <Text fw={700}>{viewContent?.name || viewContent?.title}</Text>
          <Text size="xs" c="dimmed">章节内容 ID：{recordId(viewContent)}</Text>
          <Text size="xs" c="dimmed">父级章节大纲 ID：{viewContent?.chapter_outline_id || ''}</Text>
          <Paper withBorder p="sm">
            <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>{viewContent?.content || '暂无内容。'}</Text>
          </Paper>
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setViewContent(null)}>关闭</Button>
            <Button variant="light" onClick={() => { setViewContent(null); openEditModal(viewContent); }}>直接修改章节内容</Button>
            <Button onClick={() => { setViewContent(null); openUpdateWorkflow(viewContent); }}>工作流修改章节内容</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={createOpened} onClose={() => setCreateOpened(false)} title="直接添加章节内容" size="lg">
        <Stack gap="sm">
          <Alert color="blue">这个入口直接写入 `prose` 子记录，不走工作流；会绑定到当前章节大纲下，并立即回查真实结果。</Alert>
          <TextInput
            label="章节内容标题"
            placeholder="例如：第一章正文（定稿）"
            value={createName}
            onChange={(event) => setCreateName(event.currentTarget.value)}
          />
          <Select
            label="世界观"
            data={worldviews.map((item) => ({ value: item.worldview_id, label: worldviewNameById[item.worldview_id] || item.worldview_id }))}
            value={createWorldviewId || null}
            onChange={(value) => setCreateWorldviewId(value || '')}
            clearable
            searchable
          />
          <Textarea
            label="章节内容正文"
            minRows={14}
            autosize
            value={createContent}
            onChange={(event) => setCreateContent(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setCreateOpened(false)}>取消</Button>
            <Button loading={saving} onClick={confirmCreate}>直接创建</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={editOpened} onClose={() => setEditOpened(false)} title="直接修改章节内容" size="lg">
        <Stack gap="sm">
          <Alert color="blue">这个入口直接写入 `prose` 子记录，不走工作流；保存后会立刻回查数据库结果。</Alert>
          <Text size="xs" c="dimmed">章节内容 ID：{recordId(selectedContent)}</Text>
          <TextInput
            label="章节内容标题"
            placeholder="例如：第一章正文（定稿）"
            value={editName}
            onChange={(event) => setEditName(event.currentTarget.value)}
          />
          <Select
            label="世界观"
            data={worldviews.map((item) => ({ value: item.worldview_id, label: worldviewNameById[item.worldview_id] || item.worldview_id }))}
            value={editWorldviewId || null}
            onChange={(value) => setEditWorldviewId(value || '')}
            clearable
            searchable
          />
          <Textarea
            label="章节内容正文"
            minRows={14}
            autosize
            value={editContent}
            onChange={(event) => setEditContent(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setEditOpened(false)}>取消</Button>
            <Button loading={saving} onClick={confirmUpdate}>直接保存</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={Boolean(deleteContent)} onClose={() => setDeleteContent(null)} title="删除章节内容" size="md">
        <Stack gap="sm">
          <Alert color="red">删除会移除当前章节大纲下的这条章节内容记录。</Alert>
          <Text fw={700}>{deleteContent?.name || deleteContent?.title}</Text>
          <Text size="xs" c="dimmed">{recordId(deleteContent)}</Text>
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setDeleteContent(null)}>取消</Button>
            <Button color="red" loading={saving} onClick={confirmDelete}>确认删除</Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
};
