import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  Alert,
  Badge,
  Box,
  Button,
  Divider,
  Grid,
  Group,
  Loader,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconBook,
  IconDeviceFloppy,
  IconPlus,
  IconRefresh,
  IconSettings,
  IconShieldCheck,
  IconTrash,
} from '@tabler/icons-react';
import { api, getApiErrorMessage, isApiNotFoundError } from '../api/client';

type World = {
  world_id: string;
  name: string;
};

type NovelDetailData = {
  novel_id: string;
  world_id: string;
  name: string;
  introduction?: string;
  summary?: string;
  forbidden_rules?: unknown;
  basic_settings?: unknown;
  created_at?: string;
  updated_at?: string;
};

type SettingRow = {
  id: string;
  key: string;
  value: string;
};

const uniqueWorldsById = (items: World[]): World[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (!item.world_id || seen.has(item.world_id)) return false;
    seen.add(item.world_id);
    return true;
  });
};

const normalizeRules = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    return value.split('\n').map((item) => item.trim()).filter(Boolean);
  }
  return [];
};

const normalizeSettings = (value: unknown): SettingRow[] => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value as Record<string, unknown>).map(([key, item], index) => ({
    id: `${key}-${index}`,
    key,
    value: typeof item === 'string' ? item : JSON.stringify(item, null, 2),
  }));
};

const parseSettingValue = (value: string): unknown => {
  const trimmed = value.trim();
  if (!trimmed) return '';
  try {
    return JSON.parse(trimmed);
  } catch {
    return value;
  }
};

export const NovelDetail: React.FC = () => {
  const { novelId = '' } = useParams();
  const navigate = useNavigate();
  const [worlds, setWorlds] = useState<World[]>([]);
  const [novel, setNovel] = useState<NovelDetailData | null>(null);
  const [worldId, setWorldId] = useState('');
  const [name, setName] = useState('');
  const [introduction, setIntroduction] = useState('');
  const [summary, setSummary] = useState('');
  const [rules, setRules] = useState<string[]>([]);
  const [newRule, setNewRule] = useState('');
  const [settings, setSettings] = useState<SettingRow[]>([]);
  const [newSettingKey, setNewSettingKey] = useState('');
  const [newSettingValue, setNewSettingValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [missingNovel, setMissingNovel] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);

  const loadWorlds = useCallback(async () => {
    const response = await api.listWorlds();
    setWorlds(uniqueWorldsById((response.data as World[]) || []));
  }, []);

  const loadNovel = useCallback(async () => {
    if (!novelId) return;
    setLoading(true);
    setError(null);
    setMissingNovel(false);
    setSuccess(null);
    try {
      await loadWorlds();
      const response = await api.getNovel({ novel_id: novelId });
      const nextNovel = response.data.novel as NovelDetailData;
      setNovel(nextNovel);
      setWorldId(nextNovel.world_id || '');
      setName(nextNovel.name || '');
      setIntroduction(nextNovel.introduction || '');
      setSummary(nextNovel.summary || '');
      setRules(normalizeRules(nextNovel.forbidden_rules));
      setSettings(normalizeSettings(nextNovel.basic_settings));
    } catch (err: unknown) {
      const nextMissingNovel = isApiNotFoundError(err, 'Novel not found');
      setMissingNovel(nextMissingNovel);
      setError(
        nextMissingNovel
          ? `小说 ${novelId} 不存在，可能已被删除。请返回小说列表重新选择。`
          : getApiErrorMessage(err, '加载小说详情失败。'),
      );
      setNovel(null);
    } finally {
      setLoading(false);
    }
  }, [loadWorlds, novelId]);

  useEffect(() => {
    loadNovel();
  }, [loadNovel]);

  const basicSettings = useMemo(() => {
    const next: Record<string, unknown> = {};
    settings.forEach((item) => {
      const key = item.key.trim();
      if (key) {
        next[key] = parseSettingValue(item.value);
      }
    });
    return next;
  }, [settings]);

  const addRule = () => {
    const value = newRule.trim();
    if (!value) return;
    setRules((current) => [...current, value]);
    setNewRule('');
  };

  const addSetting = () => {
    const key = newSettingKey.trim();
    if (!key) return;
    setSettings((current) => [
      ...current.filter((item) => item.key.trim() !== key),
      { id: `${key}-${Date.now()}`, key, value: newSettingValue },
    ]);
    setNewSettingKey('');
    setNewSettingValue('');
  };

  const openNovelAgentWorkflow = () => {
    if (!novelId) return;
    const params = new URLSearchParams({
      action: 'update',
      world_id: worldId,
      id: novelId,
      name: name.trim(),
      introduction,
      summary,
      forbidden_rules: JSON.stringify(rules.map((item) => item.trim()).filter(Boolean)),
      basic_settings: JSON.stringify(basicSettings),
      message: '通过 novel_agent 修改小说详情、小说禁止规则与小说设定规则',
    });
    navigate(`/workflow/novel?${params.toString()}`);
  };

  const saveNovel = async () => {
    if (!novelId || !worldId || !name.trim()) {
      setError('保存小说必须选择所属世界并填写小说名称。');
      setSuccess(null);
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateNovel({
        novel_id: novelId,
        world_id: worldId,
        name: name.trim(),
        introduction,
        summary,
        forbidden_rules: rules.map((item) => item.trim()).filter(Boolean),
        basic_settings: basicSettings,
      });
      await loadNovel();
      setSuccess(`小说已保存：${novelId}`);
    } catch (err) {
      setError(getApiErrorMessage(err, '保存小说失败。'));
    } finally {
      setSaving(false);
    }
  };

  const deleteNovel = async () => {
    if (!novelId) return;
    const confirmed = window.confirm('确认删除该小说以及旗下所有大纲、章节和相关内容？此操作不可恢复。');
    if (!confirmed) return;
    setDeleting(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteNovel({ novel_id: novelId, cascade: true });
      navigate('/novels');
    } catch (err) {
      setError(getApiErrorMessage(err, '删除小说失败。'));
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Stack gap="md" style={{ minHeight: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Group gap="xs">
            <IconBook size={24} />
            <Title order={2}>小说详情</Title>
            {novel && <Badge variant="light">{novel.novel_id}</Badge>}
          </Group>
          <Text size="sm" c="dimmed">查看小说详细信息，并维护小说禁止规则与小说设定规则。</Text>
        </Box>
        <Group gap="xs">
          <Button variant="light" leftSection={<IconArrowLeft size={16} />} onClick={() => navigate('/novels')}>返回小说列表</Button>
          <Button variant="light" leftSection={<IconRefresh size={16} />} loading={loading} onClick={loadNovel}>刷新</Button>
          <Button
            color="red"
            variant="light"
            leftSection={<IconTrash size={16} />}
            onClick={deleteNovel}
            loading={deleting}
            disabled={!novel || saving}
          >
            删除小说
          </Button>
          <Button
            color="green"
            leftSection={<IconDeviceFloppy size={16} />}
            onClick={saveNovel}
            loading={saving}
            disabled={!novel || !worldId || !name.trim() || deleting}
          >
            保存修改
          </Button>
          <Button leftSection={<IconDeviceFloppy size={16} />} onClick={openNovelAgentWorkflow} disabled={!novel || !worldId || !name.trim()}>进入小说 Agent 工作流</Button>
        </Group>
      </Group>

      {error && (
        <Alert color={missingNovel ? 'yellow' : 'red'} title={missingNovel ? '目标小说不存在或已删除' : '真实接口请求失败'}>
          {error}
        </Alert>
      )}
      {success && <Alert color="green">{success}</Alert>}
      {loading && !novel && <Loader />}

      {novel && (
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 5 }}>
            <Paper p="md" withBorder>
              <Stack gap="md">
                <Group gap="xs">
                  <IconBook size={18} />
                  <Text fw={800}>小说详细</Text>
                </Group>
                <TextInput label="小说 ID" value={novel.novel_id} readOnly />
                <Select
                  label="所属世界"
                  data={worlds.map((world) => ({ value: world.world_id, label: `${world.name} (${world.world_id})` }))}
                  value={worldId || null}
                  onChange={(value) => setWorldId(value || '')}
                  searchable
                />
                <TextInput label="小说名称" value={name} onChange={(event) => setName(event.currentTarget.value)} />
                <Textarea label="小说介绍" minRows={4} value={introduction} onChange={(event) => setIntroduction(event.currentTarget.value)} />
                <Textarea label="小说简介" minRows={8} value={summary} onChange={(event) => setSummary(event.currentTarget.value)} />
                <Divider />
                <Text size="xs" c="dimmed">创建时间：{novel.created_at || '未知'}</Text>
                <Text size="xs" c="dimmed">更新时间：{novel.updated_at || '未知'}</Text>
              </Stack>
            </Paper>
          </Grid.Col>

          <Grid.Col span={{ base: 12, md: 7 }}>
            <Stack gap="md">
              <Paper p="md" withBorder>
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Group gap="xs">
                      <IconShieldCheck size={18} />
                      <Text fw={800}>小说禁止规则</Text>
                    </Group>
                    <Badge variant="light">{rules.length} 条</Badge>
                  </Group>
                  <ScrollArea h={220}>
                    <Stack gap="xs">
                      {rules.map((rule, index) => (
                        <Textarea
                          key={`rule-${index}`}
                          label={`禁止规则 ${index + 1}`}
                          minRows={2}
                          value={rule}
                          onChange={(event) => setRules((current) => current.map((item, itemIndex) => (itemIndex === index ? event.currentTarget.value : item)))}
                        />
                      ))}
                      {rules.length === 0 && <Text size="sm" c="dimmed">当前小说还没有禁止规则。</Text>}
                    </Stack>
                  </ScrollArea>
                  <Textarea label="新增禁止规则" minRows={2} value={newRule} onChange={(event) => setNewRule(event.currentTarget.value)} />
                  <Button variant="light" leftSection={<IconPlus size={16} />} onClick={addRule}>新增禁止规则</Button>
                </Stack>
              </Paper>

              <Paper p="md" withBorder>
                <Stack gap="sm">
                  <Group justify="space-between">
                    <Group gap="xs">
                      <IconSettings size={18} />
                      <Text fw={800}>小说设定规则</Text>
                    </Group>
                    <Badge variant="light">{settings.length} 项</Badge>
                  </Group>
                  <ScrollArea h={260}>
                    <Stack gap="xs">
                      {settings.map((setting, index) => (
                        <Grid key={setting.id} gutter="xs" align="flex-start">
                          <Grid.Col span={{ base: 12, md: 4 }}>
                            <TextInput
                              label={`设定键 ${index + 1}`}
                              value={setting.key}
                              onChange={(event) => setSettings((current) => current.map((item) => (item.id === setting.id ? { ...item, key: event.currentTarget.value } : item)))}
                            />
                          </Grid.Col>
                          <Grid.Col span={{ base: 12, md: 8 }}>
                            <Textarea
                              label="设定值"
                              minRows={2}
                              value={setting.value}
                              onChange={(event) => setSettings((current) => current.map((item) => (item.id === setting.id ? { ...item, value: event.currentTarget.value } : item)))}
                            />
                          </Grid.Col>
                        </Grid>
                      ))}
                      {settings.length === 0 && <Text size="sm" c="dimmed">当前小说还没有设定规则。</Text>}
                    </Stack>
                  </ScrollArea>
                  <Grid gutter="xs" align="flex-end">
                    <Grid.Col span={{ base: 12, md: 4 }}>
                      <TextInput label="新增设定键" value={newSettingKey} onChange={(event) => setNewSettingKey(event.currentTarget.value)} />
                    </Grid.Col>
                    <Grid.Col span={{ base: 12, md: 8 }}>
                      <Textarea label="新增设定值" minRows={2} value={newSettingValue} onChange={(event) => setNewSettingValue(event.currentTarget.value)} />
                    </Grid.Col>
                  </Grid>
                  <Button variant="light" leftSection={<IconPlus size={16} />} onClick={addSetting}>新增或覆盖设定规则</Button>
                </Stack>
              </Paper>
            </Stack>
          </Grid.Col>
        </Grid>
      )}
    </Stack>
  );
};
