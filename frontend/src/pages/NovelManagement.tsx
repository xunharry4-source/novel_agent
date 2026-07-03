import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
} from '@mantine/core';
import {
  IconBook,
  IconClipboardList,
  IconEdit,
  IconFileText,
  IconListTree,
  IconPlus,
  IconRefresh,
  IconSearch,
  IconTrash,
  IconX,
} from '@tabler/icons-react';
import { api } from '../api/client';

type World = {
  world_id: string;
  name: string;
  summary?: string;
};

type Novel = {
  novel_id: string;
  world_id: string;
  name: string;
  introduction?: string;
  summary?: string;
  forbidden_rules?: string[];
  basic_settings?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
};

type NovelForm = {
  novel_id: string;
  world_id: string;
  name: string;
  introduction: string;
  summary: string;
};

const emptyForm: NovelForm = {
  novel_id: '',
  world_id: '',
  name: '',
  introduction: '',
  summary: '',
};

const uniqueWorldsById = (items: World[]): World[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (!item.world_id || seen.has(item.world_id)) return false;
    seen.add(item.world_id);
    return true;
  });
};

export const NovelManagement: React.FC = () => {
  const navigate = useNavigate();
  const [worlds, setWorlds] = useState<World[]>([]);
  const [novels, setNovels] = useState<Novel[]>([]);
  const [selectedWorldId, setSelectedWorldId] = useState('');
  const [searchText, setSearchText] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingNovel, setEditingNovel] = useState<Novel | null>(null);
  const [deleteNovel, setDeleteNovel] = useState<Novel | null>(null);
  const [cascadeDelete, setCascadeDelete] = useState(false);
  const [form, setForm] = useState<NovelForm>(emptyForm);

  const selectedWorld = useMemo(
    () => worlds.find((world) => world.world_id === selectedWorldId),
    [selectedWorldId, worlds],
  );

  const loadWorlds = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.listWorlds();
      const nextWorlds = uniqueWorldsById((response.data as World[]) || []);
      setWorlds(nextWorlds);
      setSelectedWorldId((current) => current || nextWorlds[0]?.world_id || '');
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
      setWorlds([]);
      setSelectedWorldId('');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadNovels = useCallback(async () => {
    const query = searchText.trim();
    if (!selectedWorldId && !query) {
      setNovels([]);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await api.listNovels({
        world_id: selectedWorldId || undefined,
        query: query || undefined,
        page: 1,
        page_size: 100,
      });
      const nextNovels = ((response.data as Novel[]) || []).filter((novel) => !selectedWorldId || novel.world_id === selectedWorldId);
      setNovels(nextNovels);
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
      setNovels([]);
    } finally {
      setLoading(false);
    }
  }, [searchText, selectedWorldId]);

  useEffect(() => {
    loadWorlds();
  }, [loadWorlds]);

  useEffect(() => {
    loadNovels();
  }, [loadNovels]);

  const openCreate = () => {
    const params = new URLSearchParams();
    if (selectedWorldId) params.set('world_id', selectedWorldId);
    navigate(`/novels/new${params.toString() ? `?${params.toString()}` : ''}`);
  };

  const openEdit = (novel: Novel) => {
    setEditingNovel(novel);
    setForm({
      novel_id: novel.novel_id,
      world_id: novel.world_id,
      name: novel.name || '',
      introduction: novel.introduction || '',
      summary: novel.summary || '',
    });
    setError(null);
    setSuccess(null);
    setModalOpen(true);
  };

  const saveNovel = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      if (editingNovel) {
        await api.updateNovel({
          novel_id: editingNovel.novel_id,
          world_id: form.world_id,
          name: form.name.trim(),
          introduction: form.introduction,
          summary: form.summary,
        });
        await api.getNovel({ novel_id: editingNovel.novel_id });
        setSuccess('小说已修改，并完成真实接口回查。');
      } else {
        const response = await api.createNovel({
          world_id: form.world_id,
          name: form.name.trim(),
          introduction: form.introduction,
          summary: form.summary,
          forbidden_rules: [],
          basic_settings: {},
        });
        await api.getNovel({ novel_id: response.data.novel_id });
        setSuccess('小说已新增，并完成真实接口回查。');
      }
      setModalOpen(false);
      setSelectedWorldId(form.world_id);
      await loadNovels();
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
    } finally {
      setSaving(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleteNovel) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteNovel({ novel_id: deleteNovel.novel_id, cascade: cascadeDelete });
      const response = await api.listNovels({ world_id: deleteNovel.world_id, novel_id: deleteNovel.novel_id, page: 1, page_size: 10 });
      const remaining = (response.data as Novel[]) || [];
      if (remaining.some((item) => item.novel_id === deleteNovel.novel_id)) {
        throw new Error(`删除后仍能查询到小说: ${deleteNovel.novel_id}`);
      }
      setSuccess('小说已删除，并完成真实接口回查。');
      setDeleteNovel(null);
      setCascadeDelete(false);
      await loadNovels();
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Stack gap="md" style={{ height: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Group gap="xs">
            <IconBook size={24} />
            <Title order={2}>小说管理</Title>
          </Group>
          <Text size="sm" c="dimmed">按世界筛选小说列表；点击小说进入详情页维护禁止规则与设定规则。</Text>
        </Box>
        <Group gap="xs">
          <Button variant="light" leftSection={<IconRefresh size={16} />} loading={loading} onClick={() => { loadWorlds(); loadNovels(); }}>刷新</Button>
          <Button leftSection={<IconPlus size={16} />} onClick={openCreate} disabled={!selectedWorldId}>新增小说</Button>
        </Group>
      </Group>

      {error && <Alert color="red" title="真实接口请求失败">{error}</Alert>}
      {success && <Alert color="green">{success}</Alert>}

      <Paper p="md" withBorder>
        <Group justify="space-between" align="flex-end">
          <Select
            label="世界筛选"
            data={worlds.map((world) => ({ value: world.world_id, label: `${world.name} (${world.world_id})` }))}
            value={selectedWorldId || null}
            onChange={(value) => setSelectedWorldId(value || '')}
            searchable
            clearable
            placeholder="全部世界"
            style={{ minWidth: 360 }}
          />
          <TextInput
            label="小说搜索"
            placeholder="按小说名称、ID、介绍、简介搜索"
            leftSection={<IconSearch size={16} />}
            rightSection={searchText ? <IconX size={16} style={{ cursor: 'pointer' }} onClick={() => setSearchText('')} /> : undefined}
            value={searchText}
            onChange={(event) => setSearchText(event.currentTarget.value)}
            style={{ minWidth: 360 }}
          />
          <Group gap="xs">
            {loading && <Loader size="sm" />}
            <Badge variant="light">共 {novels.length} 条</Badge>
          </Group>
        </Group>
        <Text size="xs" c="dimmed" mt="xs">
          当前筛选：{selectedWorld ? selectedWorld.summary || selectedWorld.name : '全部世界'}{searchText.trim() ? ` / 小说关键词：${searchText.trim()}` : ''}
        </Text>
      </Paper>

      <Paper p="sm" withBorder style={{ overflow: 'hidden', flex: 1 }}>
        <ScrollArea h="100%">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>小说 ID</Table.Th>
                <Table.Th>名称</Table.Th>
                <Table.Th>介绍</Table.Th>
                <Table.Th>摘要</Table.Th>
                <Table.Th>禁止规则</Table.Th>
                <Table.Th>设定规则</Table.Th>
                <Table.Th>更新时间</Table.Th>
                <Table.Th>操作</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {novels.map((novel) => (
                <Table.Tr key={novel.novel_id} onClick={() => navigate(`/novels/${encodeURIComponent(novel.novel_id)}`)} style={{ cursor: 'pointer' }} title="点击进入小说详情">
                  <Table.Td><Text size="xs" truncate maw={170}>{novel.novel_id}</Text></Table.Td>
                  <Table.Td><Text fw={700}>{novel.name}</Text></Table.Td>
                  <Table.Td><Text size="sm" truncate maw={320}>{novel.introduction || ''}</Text></Table.Td>
                  <Table.Td><Text size="sm" truncate maw={420}>{novel.summary || ''}</Text></Table.Td>
                  <Table.Td><Badge variant="light">{novel.forbidden_rules?.length || 0}</Badge></Table.Td>
                  <Table.Td><Badge variant="light">{Object.keys(novel.basic_settings || {}).length}</Badge></Table.Td>
                  <Table.Td><Text size="xs" c="dimmed">{novel.updated_at || ''}</Text></Table.Td>
                  <Table.Td>
                    <Group gap="xs" wrap="nowrap">
                      <Button size="xs" variant="light" leftSection={<IconEdit size={14} />} onClick={(event) => { event.stopPropagation(); openEdit(novel); }}>修改</Button>
                      <Button size="xs" variant="light" leftSection={<IconClipboardList size={14} />} onClick={(event) => { event.stopPropagation(); navigate(`/novels/${encodeURIComponent(novel.novel_id)}/chapter-outline-templates`); }}>大纲模板</Button>
                      <Button size="xs" variant="light" leftSection={<IconListTree size={14} />} onClick={(event) => { event.stopPropagation(); navigate(`/novels/${encodeURIComponent(novel.novel_id)}/outlines`); }}>大纲管理</Button>
                      <Button size="xs" variant="light" leftSection={<IconFileText size={14} />} onClick={(event) => { event.stopPropagation(); navigate(`/novels/${encodeURIComponent(novel.novel_id)}/chapters`); }}>章节管理</Button>
                      <Button size="xs" color="red" variant="light" leftSection={<IconTrash size={14} />} onClick={(event) => { event.stopPropagation(); setDeleteNovel(novel); }}>删除</Button>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
              {novels.length === 0 && (
                <Table.Tr>
                  <Table.Td colSpan={8}>
                    <Text ta="center" c="dimmed" py="xl">当前世界下暂无小说。</Text>
                  </Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Paper>

      <Modal opened={modalOpen} onClose={() => setModalOpen(false)} title={editingNovel ? '修改小说' : '新增小说'} size="lg">
        <Stack gap="sm">
          <Select
            label="所属世界"
            data={worlds.map((world) => ({ value: world.world_id, label: `${world.name} (${world.world_id})` }))}
            value={form.world_id || null}
            onChange={(value) => setForm({ ...form, world_id: value || '' })}
            searchable
          />
          {editingNovel && <TextInput label="小说 ID" value={editingNovel.novel_id} readOnly />}
          <TextInput label="小说名称" value={form.name} onChange={(event) => setForm({ ...form, name: event.currentTarget.value })} />
          <Textarea label="小说介绍" minRows={3} value={form.introduction} onChange={(event) => setForm({ ...form, introduction: event.currentTarget.value })} />
          <Textarea label="小说简介" minRows={5} value={form.summary} onChange={(event) => setForm({ ...form, summary: event.currentTarget.value })} />
          <Button loading={saving} onClick={saveNovel} disabled={!form.world_id || !form.name.trim()}>
            {editingNovel ? '保存修改' : '新增小说'}
          </Button>
        </Stack>
      </Modal>

      <Modal opened={Boolean(deleteNovel)} onClose={() => setDeleteNovel(null)} title="删除小说" size="md">
        <Stack gap="sm">
          <Alert color="red">删除会移除小说记录；如果勾选级联删除，还会删除该小说下的大纲与章节。</Alert>
          <Text fw={700}>{deleteNovel?.name}</Text>
          <Text size="xs" c="dimmed">{deleteNovel?.novel_id}</Text>
          <Checkbox label="级联删除大纲与章节" checked={cascadeDelete} onChange={(event) => setCascadeDelete(event.currentTarget.checked)} />
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setDeleteNovel(null)}>取消</Button>
            <Button color="red" loading={saving} onClick={confirmDelete}>确认删除</Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
};
