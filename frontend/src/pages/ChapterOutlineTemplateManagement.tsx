import React, { useCallback, useEffect, useState } from 'react';
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
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  Title,
  Select,
} from '@mantine/core';
import {
  IconArrowLeft,
  IconEdit,
  IconPlus,
  IconRefresh,
  IconTrash,
  IconClipboardList,
} from '@tabler/icons-react';
import { api, getApiErrorMessage } from '../api/client';

type Novel = {
  novel_id: string;
  world_id: string;
  name: string;
  summary?: string;
};

type ChapterOutlineTemplate = {
  template_id: string;
  novel_id: string;
  name: string;
  content: string;
  created_at?: string;
  updated_at?: string;
};

const PRESET_TEMPLATES = [
  {
    value: 'hardcore_control_chain',
    label: '⚙️ 单章硬核控制链大纲模板（全要素锁死版）',
    name: '单章硬核控制链大纲模板（全要素锁死版）',
    content: `# ⚙️ 【第X章：章名】单章硬核控制链大纲模板（全要素锁死版）

## 📑 零、 世界动态看板（World State Ledger - 本章初始输入）
> **审计规则**：本章动笔前，必须从此看板读取上章结算的最新状态。AI 扩写时，角色的语气、环境的描写必须严格锚定下列数值，禁止产生状态幻觉。

### 0. 章节总时钟锁 (Global Chapter Clock)
* **本章开场绝对时间轴**：[如：航程第 12 天，标准时 14:30:00]
* **本章预计总跨度**：[如：45 分钟 / 3 天]

### 1. 人物性格与心理状态机 (Character State Matrix)
| 角色ID | 角色名称 | 核心性格坐标 | 压力值 (0-100) | 当前心理锚点 (最在意的事) | 隐性倾向漂移 (扩写权重) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P01** | [填空] | [秩序守望/利益至上等] | [数值] | [填空] | [填空] |

## 🎬 一、 核心戏剧流与关键场景卡（Dramatic Flow & Scene Cards）
### 场景一：[场景名，如：驾驶舱的权力对峙]
* **时钟锚定**：[时间起止点]
* **戏剧功能**：[信息揭露 / 人物交锋 / 悬念设置]
* **动作线**：
  - [ ] 动作一
  - [ ] 动作二
* **心理变化因果链**：[前置原因] -> [触发事件] -> [心智转变]
`
  },
  {
    value: 'three_act_structure',
    label: '🎭 标准三幕式大纲模板',
    name: '标准三幕式大纲模板',
    content: `# 🎭 【第X章：章名】标准三幕式大纲模板

## 1. 第一幕：引入与开端 (Setup)
* **场景概述**：[描述引入本章主线冲突的开场情境]
* **出场角色**：[角色列表]
* **核心事件**：[触发冲突的具体动作]
* **悬念/冲突点**：[本幕结束时的未知或对立]

## 2. 第二幕：对抗与冲突升级 (Confrontation)
* **场景概述**：[冲突展开和深化的过程]
* **出场角色**：[角色列表]
* **核心对抗线**：[角色之间的矛盾，或角色与环境的矛盾]
* **关键动作链**：
  - [ ] 关键点一
  - [ ] 关键点二
* **转折点 (Midpoint)**：[局势发生重大变化的时刻]

## 3. 第三幕：高潮与结算 (Resolution)
* **场景概述**：[本章冲突达到顶峰并得出阶段性结果的场景]
* **出场角色**：[角色列表]
* **冲突解决方式**：[如何应对冲突，是否留下新的悬念]
* **后续影响/时钟结算**：[对下章状态的初始输入产生什么影响]
`
  },
  {
    value: 'qichengzhuanhe',
    label: '✒️ 极简起承转合大纲模板',
    name: '极简起承转合大纲模板',
    content: `# ✒️ 【第X章：章名】起承转合大纲模板

## 1. 【起】（引入）
* **本章初始状态**：[时间/地点/状态说明]
* **起因事件**：[引起事件的引子描述]

## 2. 【承】（展开）
* **情节推进**：[顺应起因展开的行动与对话]
* **细节铺陈**：[环境与人物细节补充]

## 3. 【转】（高潮）
* **戏剧性转折**：[突发状况或核心矛盾爆发]
* **情绪高潮**：[人物冲突升级]

## 4. 【合】（结算）
* **本章结局**：[事件的阶段性收尾]
* **下章悬念/线索**：[预留给下章的承接点]
`
  }
];

const pageSize = 20;

export const ChapterOutlineTemplateManagement: React.FC = () => {
  const { novelId = '' } = useParams();
  const navigate = useNavigate();
  const [novel, setNovel] = useState<Novel | null>(null);
  const [templates, setTemplates] = useState<ChapterOutlineTemplate[]>([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Modals state
  const [createOpened, setCreateOpened] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createContent, setCreateContent] = useState('');

  const [editOpened, setEditOpened] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<ChapterOutlineTemplate | null>(null);
  const [editName, setEditName] = useState('');
  const [editContent, setEditContent] = useState('');

  const [deleteTemplate, setDeleteTemplate] = useState<ChapterOutlineTemplate | null>(null);

  const hasNextPage = templates.length === pageSize;
  const paginationTotal = Math.max(page + (hasNextPage ? 1 : 0), 1);

  const loadData = useCallback(async () => {
    if (!novelId) return;
    setLoading(true);
    setError(null);
    try {
      const novelRes = await api.getNovel({ novel_id: novelId });
      setNovel(novelRes.data.novel as Novel);
      
      const templatesRes = await api.listChapterOutlineTemplates({
        novel_id: novelId,
        page,
        page_size: pageSize,
      });
      setTemplates((templatesRes.data as ChapterOutlineTemplate[]) || []);
    } catch (err: any) {
      setError(getApiErrorMessage(err));
      setTemplates([]);
    } finally {
      setLoading(false);
    }
  }, [novelId, page]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreate = async () => {
    if (!novelId || !createName.trim() || !createContent.trim()) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.createChapterOutlineTemplate({
        novel_id: novelId,
        name: createName.trim(),
        content: createContent.trim(),
      });
      setSuccess('成功创建章节大纲模板。');
      setCreateOpened(false);
      setCreateName('');
      setCreateContent('');
      setPage(1);
      await loadData();
    } catch (err: any) {
      setError(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = async () => {
    if (!editingTemplate || !editName.trim() || !editContent.trim()) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.updateChapterOutlineTemplate({
        template_id: editingTemplate.template_id,
        name: editName.trim(),
        content: editContent.trim(),
      });
      setSuccess('成功修改章节大纲模板。');
      setEditOpened(false);
      setEditingTemplate(null);
      setEditName('');
      setEditContent('');
      await loadData();
    } catch (err: any) {
      setError(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTemplate) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await api.deleteChapterOutlineTemplate({
        template_id: deleteTemplate.template_id,
      });
      setSuccess('成功删除章节大纲模板。');
      setDeleteTemplate(null);
      await loadData();
    } catch (err: any) {
      setError(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const openEdit = (template: ChapterOutlineTemplate) => {
    setEditingTemplate(template);
    setEditName(template.name);
    setEditContent(template.content);
    setEditOpened(true);
  };

  const handlePresetSelect = (value: string | null) => {
    if (!value) return;
    const preset = PRESET_TEMPLATES.find((p) => p.value === value);
    if (preset) {
      setCreateName(preset.name);
      setCreateContent(preset.content);
    }
  };

  return (
    <Stack gap="md" style={{ height: 'calc(100vh - 96px)' }}>
      <Group justify="space-between" align="flex-start">
        <Box>
          <Group gap="xs" mb="xs">
            <Button
              variant="subtle"
              leftSection={<IconArrowLeft size={16} />}
              onClick={() => navigate('/novels')}
              p={0}
            >
              返回小说管理
            </Button>
          </Group>
          <Group gap="xs">
            <IconClipboardList size={24} />
            <Title order={2}>章节大纲模板管理</Title>
            {novel && (
              <Badge size="lg" color="blue">
                小说：{novel.name}
              </Badge>
            )}
          </Group>
          <Text size="sm" c="dimmed">
            在这里，你可以为当前小说定制不同的章节大纲生成模板（如三幕式结构、英雄旅程等）。
          </Text>
        </Box>
        <Group gap="xs">
          <Button
            variant="light"
            leftSection={<IconRefresh size={16} />}
            loading={loading}
            onClick={loadData}
          >
            刷新
          </Button>
          <Button
            leftSection={<IconPlus size={16} />}
            onClick={() => setCreateOpened(true)}
          >
            新增大纲模板
          </Button>
        </Group>
      </Group>

      {error && (
        <Alert color="red" title="请求失败">
          {error}
        </Alert>
      )}
      {success && <Alert color="green">{success}</Alert>}

      <Paper p="sm" withBorder style={{ overflow: 'hidden', flex: 1 }}>
        <ScrollArea h="100%">
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>模板 ID</Table.Th>
                <Table.Th>模板名称</Table.Th>
                <Table.Th>内容预览</Table.Th>
                <Table.Th>更新时间</Table.Th>
                <Table.Th>操作</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {templates.map((tpl) => (
                <Table.Tr key={tpl.template_id}>
                  <Table.Td>
                    <Text size="xs" truncate maw={120}>
                      {tpl.template_id}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text fw={700}>{tpl.name}</Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" truncate maw={400}>
                      {tpl.content}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {tpl.updated_at || ''}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Group gap="xs" wrap="nowrap">
                      <Button
                        size="xs"
                        variant="light"
                        leftSection={<IconEdit size={14} />}
                        onClick={() => openEdit(tpl)}
                      >
                        修改
                      </Button>
                      <Button
                        size="xs"
                        color="red"
                        variant="light"
                        leftSection={<IconTrash size={14} />}
                        onClick={() => setDeleteTemplate(tpl)}
                      >
                        删除
                      </Button>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
              {templates.length === 0 && (
                <Table.Tr>
                  <Table.Td colSpan={5}>
                    <Text ta="center" c="dimmed" py="xl">
                      当前小说下暂无大纲模板。
                    </Text>
                  </Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Paper>

      {templates.length > 0 && (
        <Group justify="center">
          <Pagination
            value={page}
            onChange={setPage}
            total={paginationTotal}
            disabled={loading}
          />
        </Group>
      )}

      {/* Create Modal */}
      <Modal
        opened={createOpened}
        onClose={() => setCreateOpened(false)}
        title="新增章节大纲模板"
        size="xl"
      >
        <Stack gap="sm">
          <Select
            label="从预设大纲模板填充（可选）"
            placeholder="选择一个经典预设结构快速填充内容..."
            data={PRESET_TEMPLATES}
            onChange={handlePresetSelect}
            clearable
          />
          <TextInput
            label="模板名称"
            placeholder="例如：三幕式结构"
            required
            value={createName}
            onChange={(event) => setCreateName(event.currentTarget.value)}
          />
          <Textarea
            label="模板内容"
            placeholder="输入大纲模板的具体结构或描述（支持 Markdown）"
            required
            autosize
            minRows={20}
            maxRows={35}
            value={createContent}
            onChange={(event) => setCreateContent(event.currentTarget.value)}
          />
          <Button
            loading={saving}
            onClick={handleCreate}
            disabled={!createName.trim() || !createContent.trim()}
          >
            保存模板
          </Button>
        </Stack>
      </Modal>

      {/* Edit Modal */}
      <Modal
        opened={editOpened}
        onClose={() => setEditOpened(false)}
        title="修改章节大纲模板"
        size="xl"
      >
        <Stack gap="sm">
          <TextInput
            label="模板名称"
            required
            value={editName}
            onChange={(event) => setEditName(event.currentTarget.value)}
          />
          <Textarea
            label="模板内容"
            required
            autosize
            minRows={20}
            maxRows={35}
            value={editContent}
            onChange={(event) => setEditContent(event.currentTarget.value)}
          />
          <Button
            loading={saving}
            onClick={handleEdit}
            disabled={!editName.trim() || !editContent.trim()}
          >
            保存修改
          </Button>
        </Stack>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        opened={Boolean(deleteTemplate)}
        onClose={() => setDeleteTemplate(null)}
        title="删除章节大纲模板"
        size="md"
      >
        <Stack gap="sm">
          <Alert color="red">确定要删除该章节大纲模板吗？此操作不可逆。</Alert>
          <Text fw={700}>{deleteTemplate?.name}</Text>
          <Text size="xs" c="dimmed">
            {deleteTemplate?.template_id}
          </Text>
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setDeleteTemplate(null)}>
              取消
            </Button>
            <Button color="red" loading={saving} onClick={handleDelete}>
              确认删除
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
};
export default ChapterOutlineTemplateManagement;
