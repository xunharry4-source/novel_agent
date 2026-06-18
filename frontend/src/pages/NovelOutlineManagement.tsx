import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
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
  IconEdit,
  IconFileText,
  IconListTree,
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
  summary?: string;
};

type Worldview = {
  worldview_id: string;
  name?: string;
  title?: string;
};

type Outline = {
  outline_id: string;
  id?: string;
  novel_id?: string;
  world_id?: string;
  worldview_id?: string;
  name?: string;
  title?: string;
  summary?: string;
  updated_at?: string;
  created_at?: string;
};

const pageSize = 20;

export const NovelOutlineManagement: React.FC = () => {
  const { novelId = '' } = useParams();
  const navigate = useNavigate();
  const [novel, setNovel] = useState<Novel | null>(null);
  const [worldviews, setWorldviews] = useState<Worldview[]>([]);
  const [outlines, setOutlines] = useState<Outline[]>([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [missingNovel, setMissingNovel] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [createOpened, setCreateOpened] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createSummary, setCreateSummary] = useState('');
  const [createWorldviewId, setCreateWorldviewId] = useState('');
  const [editOpened, setEditOpened] = useState(false);
  const [editingOutline, setEditingOutline] = useState<Outline | null>(null);
  const [editName, setEditName] = useState('');
  const [editSummary, setEditSummary] = useState('');
  const [editWorldviewId, setEditWorldviewId] = useState('');
  const [deleteOutline, setDeleteOutline] = useState<Outline | null>(null);

  const hasNextPage = outlines.length === pageSize;
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
        api.listOutlines({ novel_id: novelId, page, page_size: pageSize }),
        api.listWorldviews({ world_id: nextNovel.world_id, page: 1, page_size: 100 }),
      ]);
      const nextOutlines = ((outlineRes.data as Outline[]) || []).filter((outline) => outline.novel_id === novelId);
      setNovel(nextNovel);
      setWorldviews((worldviewRes.data as Worldview[]) || []);
      setOutlines(nextOutlines);
    } catch (err: unknown) {
      const nextMissingNovel = isApiNotFoundError(err, 'Novel not found');
      setMissingNovel(nextMissingNovel);
      setError(
        nextMissingNovel
          ? `小说 ${novelId} 不存在，可能已被删除。请返回小说列表重新选择。`
          : getApiErrorMessage(err, '加载分卷大纲失败。'),
      );
      setNovel(null);
      setWorldviews([]);
      setOutlines([]);
    } finally {
      setLoading(false);
    }
  }, [novelId, page]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const openCreateWorkflow = () => {
    if (!novel) return;
    const params = new URLSearchParams({
      action: 'create',
      world_id: novel.world_id,
      novel_id: novel.novel_id,
    });
    navigate(`/workflow/outline?${params.toString()}`);
  };

  const openCreateModal = () => {
    if (!novel) return;
    setCreateName('');
    setCreateSummary('');
    setCreateWorldviewId((current) => current || worldviews[0]?.worldview_id || '');
    setCreateOpened(true);
  };

  const openUpdateWorkflow = (outline: Outline) => {
    if (!novel) return;
    const outlineId = outline.outline_id || outline.id || '';
    const params = new URLSearchParams({
      action: 'update',
      world_id: outline.world_id || novel.world_id,
      novel_id: novel.novel_id,
      id: outlineId,
      name: outline.name || outline.title || '',
      summary: outline.summary || '',
    });
    if (outline.worldview_id) params.set('worldview_id', outline.worldview_id);
    navigate(`/workflow/outline?${params.toString()}`);
  };

  const openEditModal = (outline: Outline) => {
    setEditingOutline(outline);
    setEditName(outline.name || outline.title || '');
    setEditSummary(outline.summary || '');
    setEditWorldviewId(outline.worldview_id || worldviews[0]?.worldview_id || '');
    setEditOpened(true);
  };

  const openChapterManagement = (outline?: Outline) => {
    const params = new URLSearchParams();
    if (outline?.outline_id) params.set('outline_id', outline.outline_id);
    navigate(`/novels/${encodeURIComponent(novelId)}/chapters${params.toString() ? `?${params.toString()}` : ''}`);
  };

  const confirmDelete = async () => {
    if (!deleteOutline) return;
    const outlineId = deleteOutline.outline_id || deleteOutline.id || '';
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteArchiveItem({ id: outlineId, type: 'outline', novel_id: novelId });
      const verifyRes = await api.listOutlines({ novel_id: novelId, outline_id: outlineId, page: 1, page_size: 10 });
      const remaining = ((verifyRes.data as Outline[]) || []).filter((outline) => (outline.outline_id || outline.id) === outlineId);
      if (remaining.length > 0) {
        throw new Error(`删除后仍能查询到分卷大纲: ${outlineId}`);
      }
      setSuccess('分卷大纲已删除，并完成真实接口回查。');
      setDeleteOutline(null);
      await loadData();
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '删除分卷大纲失败。'));
    } finally {
      setSaving(false);
    }
  };

  const confirmCreate = async () => {
    if (!novel) return;
    const nextName = createName.trim();
    if (!nextName) {
      setError('分卷大纲名称不能为空。');
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const createRes = await api.createOutline({
        name: nextName,
        summary: createSummary.trim() || undefined,
        worldview_id: createWorldviewId || undefined,
        novel_id: novel.novel_id,
        world_id: novel.world_id,
      });
      const createdOutlineId = createRes.data.outline_id as string;
      const verifyRes = await api.listOutlines({ novel_id: novel.novel_id, outline_id: createdOutlineId, page: 1, page_size: 10 });
      const created = ((verifyRes.data as Outline[]) || []).find((outline) => (outline.outline_id || outline.id) === createdOutlineId);
      if (!created) {
        throw new Error(`新增后未查询到分卷大纲: ${createdOutlineId}`);
      }
      setSuccess(`分卷大纲已直接创建，并完成真实接口回查：${createdOutlineId}`);
      setCreateOpened(false);
      setCreateName('');
      setCreateSummary('');
      setCreateWorldviewId(worldviews[0]?.worldview_id || '');
      if (page !== 1) {
        setPage(1);
      } else {
        await loadData();
      }
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '新增分卷大纲失败。'));
    } finally {
      setSaving(false);
    }
  };

  const confirmUpdate = async () => {
    if (!novel || !editingOutline) return;
    const outlineId = editingOutline.outline_id || editingOutline.id || '';
    const nextName = editName.trim();
    if (!outlineId) {
      setError('缺少分卷大纲 ID，无法直接修改。');
      return;
    }
    if (!nextName) {
      setError('分卷大纲名称不能为空。');
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateArchiveItem({
        id: outlineId,
        type: 'outline',
        name: nextName,
        content: editSummary,
        world_id: editingOutline.world_id || novel.world_id,
        novel_id: novel.novel_id,
        worldview_id: editWorldviewId || undefined,
      });
      const verifyRes = await api.listOutlines({ novel_id: novel.novel_id, outline_id: outlineId, page: 1, page_size: 10 });
      const updated = ((verifyRes.data as Outline[]) || []).find((outline) => (outline.outline_id || outline.id) === outlineId);
      if (!updated) {
        throw new Error(`修改后未查询到分卷大纲: ${outlineId}`);
      }
      if ((updated.name || updated.title || '') !== nextName) {
        throw new Error(`分卷大纲名称未更新，实际为: ${updated.name || updated.title || ''}`);
      }
      if ((updated.summary || '') !== editSummary) {
        throw new Error('分卷大纲正文未更新为提交内容。');
      }
      if ((editWorldviewId || '') !== (updated.worldview_id || '')) {
        throw new Error(`分卷大纲世界观未更新，实际为: ${updated.worldview_id || ''}`);
      }
      setSuccess(`分卷大纲已直接修改，并完成真实接口回查：${outlineId}`);
      setEditOpened(false);
      setEditingOutline(null);
      await loadData();
    } catch (err: unknown) {
      setError(getApiErrorMessage(err, '修改分卷大纲失败。'));
    } finally {
      setSaving(false);
    }
  };

  const title = useMemo(() => novel?.name || novelId, [novel?.name, novelId]);
  const worldviewNameById = useMemo(
    () => Object.fromEntries(worldviews.map((worldview) => [worldview.worldview_id, getWorldviewDisplayName(worldview)])),
    [worldviews],
  );

  return (
    <Stack gap="md" style={{ height: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Group gap="xs">
            <IconListTree size={24} />
            <Title order={2}>分卷大纲管理</Title>
            <Badge variant="light">{novelId}</Badge>
          </Group>
          <Text size="sm" c="dimmed">当前小说：{title}。本页只管理分卷大纲；分卷章节大纲请进入独立章节页。</Text>
        </Box>
        <Group gap="xs">
          <Button variant="light" leftSection={<IconArrowLeft size={16} />} onClick={() => navigate('/novels')}>返回小说列表</Button>
          <Button variant="light" leftSection={<IconRefresh size={16} />} loading={loading} onClick={loadData}>刷新</Button>
          <Button variant="light" leftSection={<IconFileText size={16} />} onClick={() => openChapterManagement()}>分卷章节大纲页</Button>
          <Button variant="light" leftSection={<IconPlus size={16} />} onClick={openCreateWorkflow} disabled={!novel}>工作流新增分卷大纲</Button>
          <Button leftSection={<IconPlus size={16} />} onClick={openCreateModal} disabled={!novel}>直接新增分卷大纲</Button>
        </Group>
      </Group>

      {error && (
        <Alert color={missingNovel ? 'yellow' : 'red'} title={missingNovel ? '目标小说不存在或已删除' : '真实接口请求失败'}>
          {error}
        </Alert>
      )}
      {success && <Alert color="green">{success}</Alert>}

      <Paper p="sm" withBorder style={{ overflow: 'hidden', flex: 1 }}>
        <Group justify="space-between" mb="sm">
          <Group gap="xs">
            {loading && <Loader size="sm" />}
            <Badge variant="light">第 {page} 页</Badge>
            <Badge variant="light">本页 {outlines.length} 条</Badge>
          </Group>
          <Pagination value={page} onChange={setPage} total={paginationTotal} />
        </Group>
        <ScrollArea h="calc(100vh - 245px)">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>分卷大纲 ID</Table.Th>
                <Table.Th>卷名</Table.Th>
                <Table.Th>简介 / 摘要</Table.Th>
                <Table.Th>世界观</Table.Th>
                <Table.Th>更新时间</Table.Th>
                <Table.Th>操作</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {outlines.map((outline) => {
                const outlineId = outline.outline_id || outline.id || '';
                return (
                  <Table.Tr key={outlineId}>
                    <Table.Td><Text size="xs" truncate maw={180}>{outlineId}</Text></Table.Td>
                    <Table.Td><Text fw={700}>{outline.name || outline.title || ''}</Text></Table.Td>
                    <Table.Td><Text size="sm" truncate maw={520}>{outline.summary || ''}</Text></Table.Td>
                    <Table.Td><Text size="xs" truncate maw={180}>{outline.worldview_id ? (worldviewNameById[outline.worldview_id] || outline.worldview_id) : ''}</Text></Table.Td>
                    <Table.Td><Text size="xs" c="dimmed">{outline.updated_at || outline.created_at || ''}</Text></Table.Td>
                    <Table.Td>
                      <Group gap="xs" wrap="nowrap">
                        <Button size="xs" variant="light" leftSection={<IconFileText size={14} />} onClick={() => openChapterManagement(outline)}>章节大纲</Button>
                        <Button size="xs" variant="light" leftSection={<IconEdit size={14} />} onClick={() => openEditModal(outline)}>直接修改</Button>
                        <Button size="xs" variant="light" onClick={() => openUpdateWorkflow(outline)}>工作流修改</Button>
                        <Button size="xs" color="red" variant="light" leftSection={<IconTrash size={14} />} onClick={() => setDeleteOutline(outline)}>删除</Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
              {outlines.length === 0 && (
                <Table.Tr>
                  <Table.Td colSpan={6}>
                    <Text ta="center" c="dimmed" py="xl">当前小说下暂无分卷大纲。</Text>
                  </Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Paper>

      <Modal opened={createOpened} onClose={() => setCreateOpened(false)} title="直接新增分卷大纲" size="lg">
        <Stack gap="sm">
          <Alert color="blue">这个入口直接写入 `outlines`，不走工作流。</Alert>
          <TextInput
            label="卷名"
            placeholder="例如：第一卷：失踪者"
            value={createName}
            onChange={(event) => setCreateName(event.currentTarget.value)}
          />
          <Select
            label="世界观"
            data={worldviews.map((worldview) => ({ value: worldview.worldview_id, label: getWorldviewDisplayName(worldview) }))}
            value={createWorldviewId || null}
            onChange={(value) => setCreateWorldviewId(value || '')}
            clearable
            searchable
          />
          <Textarea
            label="大纲正文 / 简介"
            minRows={10}
            autosize
            value={createSummary}
            onChange={(event) => setCreateSummary(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setCreateOpened(false)}>取消</Button>
            <Button loading={saving} onClick={confirmCreate}>直接创建</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={editOpened} onClose={() => setEditOpened(false)} title="直接修改分卷大纲" size="lg">
        <Stack gap="sm">
          <Alert color="blue">这个入口直接写入 `outlines`，不走工作流；保存后会立刻回查数据库结果。</Alert>
          <Text size="xs" c="dimmed">分卷大纲 ID：{editingOutline?.outline_id || editingOutline?.id || ''}</Text>
          <TextInput
            label="卷名"
            placeholder="例如：第一卷：失踪者"
            value={editName}
            onChange={(event) => setEditName(event.currentTarget.value)}
          />
          <Select
            label="世界观"
            data={worldviews.map((worldview) => ({ value: worldview.worldview_id, label: getWorldviewDisplayName(worldview) }))}
            value={editWorldviewId || null}
            onChange={(value) => setEditWorldviewId(value || '')}
            clearable
            searchable
          />
          <Textarea
            label="大纲正文 / 简介"
            minRows={10}
            autosize
            value={editSummary}
            onChange={(event) => setEditSummary(event.currentTarget.value)}
          />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setEditOpened(false)}>取消</Button>
            <Button loading={saving} onClick={confirmUpdate}>直接保存</Button>
          </Group>
        </Stack>
      </Modal>

      <Modal opened={Boolean(deleteOutline)} onClose={() => setDeleteOutline(null)} title="删除分卷大纲" size="md">
        <Stack gap="sm">
          <Alert color="red">删除会移除该分卷大纲记录；如果章节仍引用该分卷，请先确认影响范围。</Alert>
          <Text fw={700}>{deleteOutline?.name || deleteOutline?.title}</Text>
          <Text size="xs" c="dimmed">{deleteOutline?.outline_id || deleteOutline?.id}</Text>
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setDeleteOutline(null)}>取消</Button>
            <Button color="red" loading={saving} onClick={confirmDelete}>确认删除</Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
};
