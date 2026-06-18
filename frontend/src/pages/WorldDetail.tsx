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
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconDeviceFloppy,
  IconPlus,
  IconRefresh,
  IconSettings,
  IconShieldCheck,
  IconTrash,
  IconWorld,
} from '@tabler/icons-react';
import { api, getApiErrorMessage } from '../api/client';

type WorldDetailData = {
  world_id: string;
  name: string;
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

export const WorldDetail: React.FC = () => {
  const { worldId = '' } = useParams();
  const navigate = useNavigate();
  const [world, setWorld] = useState<WorldDetailData | null>(null);
  const [name, setName] = useState('');
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
  const [success, setSuccess] = useState<string | null>(null);

  const loadWorld = useCallback(async () => {
    if (!worldId) return;
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const response = await api.getWorld({ world_id: worldId });
      const nextWorld = response.data.world as WorldDetailData;
      setWorld(nextWorld);
      setName(nextWorld.name || '');
      setSummary(nextWorld.summary || '');
      setRules(normalizeRules(nextWorld.forbidden_rules));
      setSettings(normalizeSettings(nextWorld.basic_settings));
    } catch (err: any) {
      setError(err?.response?.data?.error || err?.message || String(err));
      setWorld(null);
    } finally {
      setLoading(false);
    }
  }, [worldId]);

  useEffect(() => {
    loadWorld();
  }, [loadWorld]);

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

  const openWorldAgentWorkflow = () => {
    if (!worldId) return;
    const params = new URLSearchParams({
      action: 'update',
      id: worldId,
      name: name.trim(),
      summary,
      forbidden_rules: JSON.stringify(rules.map((item) => item.trim()).filter(Boolean)),
      basic_settings: JSON.stringify(basicSettings),
      message: '通过 world_agent 修改世界详情、世界禁止规则与世界设定规则',
    });
    navigate(`/workflow/world?${params.toString()}`);
  };

  const saveWorld = async () => {
    if (!worldId || !name.trim()) {
      setError('保存世界必须填写世界名称。');
      setSuccess(null);
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateWorld({
        world_id: worldId,
        name: name.trim(),
        summary,
        forbidden_rules: rules.map((item) => item.trim()).filter(Boolean),
        basic_settings: basicSettings,
      });
      await loadWorld();
      setSuccess(`世界已保存：${worldId}`);
    } catch (err) {
      setError(getApiErrorMessage(err, '保存世界失败。'));
    } finally {
      setSaving(false);
    }
  };

  const deleteWorld = async () => {
    if (!worldId) return;
    const confirmed = window.confirm('确认删除该世界以及旗下所有世界观、小说、大纲、章节和设定库内容？此操作不可恢复。');
    if (!confirmed) return;
    setDeleting(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteWorld({ world_id: worldId, cascade: true });
      navigate('/worlds');
    } catch (err) {
      setError(getApiErrorMessage(err, '删除世界失败。'));
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Stack gap="md" style={{ minHeight: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Group gap="xs">
            <IconWorld size={24} />
            <Title order={2}>世界详情</Title>
            {world && <Badge variant="light">{world.world_id}</Badge>}
          </Group>
          <Text size="sm" c="dimmed">查看世界详细信息，并维护世界禁止规则与世界设定规则。</Text>
        </Box>
        <Group gap="xs">
          <Button variant="light" leftSection={<IconArrowLeft size={16} />} onClick={() => navigate('/worlds')}>返回世界列表</Button>
          <Button variant="light" leftSection={<IconRefresh size={16} />} loading={loading} onClick={loadWorld}>刷新</Button>
          <Button
            color="red"
            variant="light"
            leftSection={<IconTrash size={16} />}
            onClick={deleteWorld}
            loading={deleting}
            disabled={!world || saving}
          >
            删除世界
          </Button>
          <Button
            color="green"
            leftSection={<IconDeviceFloppy size={16} />}
            onClick={saveWorld}
            loading={saving}
            disabled={!world || !name.trim() || deleting}
          >
            保存修改
          </Button>
          <Button leftSection={<IconDeviceFloppy size={16} />} onClick={openWorldAgentWorkflow} disabled={!world || !name.trim()}>进入世界 Agent 工作流</Button>
        </Group>
      </Group>

      {error && <Alert color="red" title="真实接口请求失败">{error}</Alert>}
      {success && <Alert color="green">{success}</Alert>}
      {loading && !world && <Loader />}

      {world && (
        <Grid gutter="md">
          <Grid.Col span={{ base: 12, md: 5 }}>
            <Paper p="md" withBorder>
              <Stack gap="md">
                <Group gap="xs">
                  <IconWorld size={18} />
                  <Text fw={800}>世界详细</Text>
                </Group>
                <TextInput label="世界 ID" value={world.world_id} readOnly />
                <TextInput label="世界名称" value={name} onChange={(event) => setName(event.currentTarget.value)} />
                <Textarea label="世界摘要" minRows={8} value={summary} onChange={(event) => setSummary(event.currentTarget.value)} />
                <Divider />
                <Text size="xs" c="dimmed">创建时间：{world.created_at || '未知'}</Text>
                <Text size="xs" c="dimmed">更新时间：{world.updated_at || '未知'}</Text>
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
                      <Text fw={800}>世界禁止规则</Text>
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
                      {rules.length === 0 && <Text size="sm" c="dimmed">当前世界还没有禁止规则。</Text>}
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
                      <Text fw={800}>世界设定规则</Text>
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
                      {settings.length === 0 && <Text size="sm" c="dimmed">当前世界还没有设定规则。</Text>}
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
