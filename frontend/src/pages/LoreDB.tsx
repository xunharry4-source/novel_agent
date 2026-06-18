import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Alert, Box, Typography, Paper,
  IconButton, Button, TextField, InputAdornment, Chip,
  List, ListItem, ListItemButton, ListItemText, ListItemIcon,
  CircularProgress, Tooltip, Stack, Collapse,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  ToggleButton, ToggleButtonGroup,
  FormControl, InputLabel, MenuItem, Select
} from '@mui/material';
import {
  Search as SearchIcon,
  Add as AddIcon,
  LibraryBooks as LoreIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  PlaylistAdd as AddSiblingIcon,
  Public as WorldviewIcon,
  Refresh as RefreshIcon,
  AccountTree as TreeIcon,
  SubdirectoryArrowRight as AddChildIcon,
  TableRows as TableIcon,
  UploadFile as UploadFileIcon,
  ExpandMore as ExpandMoreIcon,
  ChevronRight as ChevronRightIcon
} from '@mui/icons-material';
import { getWorldviewDisplayName } from '../utils/worldview';
import { useNavigate } from 'react-router-dom';

// --- Types ---
interface World {
  world_id: string;
  name: string;
  summary?: string;
}

interface Worldview {
  worldview_id: string;
  world_id?: string;
  name?: string;
  title?: string;
  summary?: string;
}

interface LoreEntry {
  id: string;
  type: string;
  name: string;
  content: string;
  category: string;
  path?: string;
  timestamp?: string;
  outline_id?: string | null;
  worldview_id?: string | null;
  world_id?: string | null;
}

interface LoreTreeNode {
  name: string;
  path: string;
  order: number;
  children: Record<string, LoreTreeNode>;
  entries: LoreEntry[];
  hierarchyPath?: string[];
}

const normalizeHierarchyPath = (rawPath: string | undefined | null): string[] => (
  String(rawPath || '')
    .replace(/\//g, '>')
    .split('>')
    .map((part) => part.trim())
    .filter(Boolean)
);

export const LoreDB: React.FC = () => {
  const [worlds, setWorlds] = useState<World[]>([]);
  const [selectedWorldId, setSelectedWorldId] = useState<string>('');
  const [worldviews, setWorldviews] = useState<Worldview[]>([]);
  const [selectedWV, setSelectedWV] = useState<string | null>(null);
  const [entries, setEntries] = useState<LoreEntry[]>([]);
  const [viewMode, setViewMode] = useState<'tree' | 'table'>('tree');
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [entryPage, setEntryPage] = useState(1);
  const [entryPageSize, setEntryPageSize] = useState(20);
  const [hasMoreEntries, setHasMoreEntries] = useState(false);
  const [importing, setImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const navigate = useNavigate();

  // --- API Calls ---

  const fetchWorlds = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/worlds/list');
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const data = await res.json();
      setWorlds(data);
      if (data.length > 0) {
        setSelectedWorldId((current) => current || data[0].world_id);
      } else {
        setSelectedWorldId('');
        setWorldviews([]);
        setSelectedWV(null);
        setEntries([]);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
      setWorlds([]);
      setSelectedWorldId('');
      setWorldviews([]);
      setSelectedWV(null);
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchWorldviews = useCallback(async (currentWorldId: string) => {
    if (!currentWorldId) {
      setWorldviews([]);
      setSelectedWV(null);
      setEntries([]);
      setHasMoreEntries(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/worldviews/list?world_id=${encodeURIComponent(currentWorldId)}&page=1&page_size=50`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const data = await res.json();
      const mismatched = data.filter((worldview: Worldview) => worldview.world_id !== currentWorldId);
      if (mismatched.length > 0) {
        throw new Error(`接口返回了非当前世界的世界观: ${mismatched.map((item: Worldview) => item.worldview_id).join(', ')}`);
      }
      setWorldviews(data);
      if (data.length > 1) {
        setSelectedWV(null);
        setEntries([]);
        setHasMoreEntries(false);
        setError(`当前世界返回了 ${data.length} 个世界观设定集，违反“一世界一设定集”规则，/worldviews 无法继续按正常结构管理设定条目。`);
        return;
      }
      setSelectedWV(data[0]?.worldview_id || null);
      if (data.length === 0) {
        setEntries([]);
        setHasMoreEntries(false);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
      setWorldviews([]);
      setSelectedWV(null);
      setEntries([]);
      setHasMoreEntries(false);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchEntries = useCallback(async (currentWorldId: string, wvId: string, query: string = '', page: number = 1, pageSize: number = 20) => {
    if (!currentWorldId || !wvId) {
      setEntries([]);
      setHasMoreEntries(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const url = `/api/lore/list?world_id=${encodeURIComponent(currentWorldId)}&worldview_id=${encodeURIComponent(wvId)}${query ? `&query=${encodeURIComponent(query)}` : ''}&page=${page}&page_size=${pageSize}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const data = await res.json();
      const mismatched = data.filter((entry: LoreEntry) => entry.world_id !== currentWorldId);
      if (mismatched.length > 0) {
        throw new Error(`接口返回了非当前世界的设定条目: ${mismatched.map((entry: LoreEntry) => entry.id).join(', ')}`);
      }
      setEntries(data);
      setHasMoreEntries(data.length === pageSize);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
      setEntries([]);
      setHasMoreEntries(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWorlds();
  }, [fetchWorlds]);

  useEffect(() => {
    setEntryPage(1);
    setExpandedNodes({});
  }, [searchQuery, selectedWorldId, selectedWV]);

  useEffect(() => {
    fetchWorldviews(selectedWorldId);
  }, [selectedWorldId, fetchWorldviews]);

  useEffect(() => {
    if (selectedWorldId && selectedWV) {
      fetchEntries(selectedWorldId, selectedWV, searchQuery, entryPage, entryPageSize);
    } else {
      setEntries([]);
      setHasMoreEntries(false);
    }
  }, [entryPage, entryPageSize, selectedWorldId, selectedWV, searchQuery, fetchEntries]);

  const handleDeleteEntry = async (id: string) => {
    if (!selectedWorldId || !selectedWV) return;
    if (!window.confirm('确定要物理清理该设定条目吗？不可恢复。')) return;
    setError(null);
    try {
      const res = await fetch('/api/archive/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, type: 'worldview', world_id: selectedWorldId, worldview_id: selectedWV })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      fetchEntries(selectedWorldId, selectedWV, searchQuery, entryPage, entryPageSize);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  };

  const openImportPicker = () => {
    if (!selectedWorldId || !selectedWV || worldviews.length !== 1) return;
    fileInputRef.current?.click();
  };

  const importWorldviewHierarchy = async (file: File) => {
    if (!selectedWorldId || !selectedWV) return;
    setImporting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('world_id', selectedWorldId);
      form.append('worldview_id', selectedWV);
      form.append('file', file);
      const response = await fetch('/api/worldviews/import', {
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
      setEntryPage(1);
      await fetchEntries(selectedWorldId, selectedWV, searchQuery, 1, entryPageSize);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      setImporting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleImportFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await importWorldviewHierarchy(file);
  };

  // --- UI Renderers ---
  const loreTree = useMemo(() => {
    const root: LoreTreeNode = { name: '全部设定', path: 'root', order: 0, children: {}, entries: [] };
    const selectedWorld = worlds.find((world) => world.world_id === selectedWorldId);
    const worldName = selectedWorld?.name || selectedWorldId || '未选择世界';
    const selectedWorldview = worldviews.find((worldview) => worldview.worldview_id === selectedWV);
    const worldviewName = getWorldviewDisplayName(selectedWorldview) || selectedWV || '未选择世界观';
    const outlineNameById = new Map<string, string>();

    entries.forEach((entry) => {
      if (entry.type === 'outline') {
        const outlineId = entry.outline_id || entry.id;
        outlineNameById.set(outlineId, entry.name || outlineId);
      }
    });

    const getChild = (
      parent: LoreTreeNode,
      key: string,
      name: string,
      order: number,
      hierarchyPath?: string[],
    ) => {
      if (!parent.children[key]) {
        parent.children[key] = {
          name,
          path: `${parent.path}/${key}`,
          order,
          children: {},
          entries: [],
          hierarchyPath,
        };
      } else {
        parent.children[key].name = name;
        if (hierarchyPath !== undefined) {
          parent.children[key].hierarchyPath = hierarchyPath;
        }
      }
      return parent.children[key];
    };

    const worldNode = getChild(root, `world:${selectedWorldId || 'unknown'}`, `世界：${worldName}`, 0);
    const worldviewNode = getChild(
      worldNode,
      `worldview:${selectedWV || 'unknown'}`,
      `世界观：${worldviewName}`,
      0
    );
    const worldviewEntriesNode = getChild(worldviewNode, 'worldview-entries', '世界观设定', 0, []);
    const uncategorizedNovelNode = getChild(worldviewNode, 'novel:unassigned', '小说：未归属小说', 99);
    const uncategorizedOutlineNode = getChild(uncategorizedNovelNode, 'outline:unassigned', '大纲：未归属大纲', 0);

    const getNovelOutlineNode = (entry: LoreEntry) => {
      const outlineId = entry.outline_id || (entry.type === 'outline' ? entry.id : null);
      if (!outlineId) return uncategorizedOutlineNode;

      const novelName = outlineNameById.get(outlineId) || outlineId;
      const novelNode = getChild(worldviewNode, `novel:${outlineId}`, `小说：${novelName}`, 10);
      return getChild(novelNode, `outline:${outlineId}`, `大纲：${novelName}`, 0);
    };

    entries.forEach((entry) => {
      if (entry.type === 'worldview') {
        const pathParts = normalizeHierarchyPath(entry.path || entry.category || '');
        const nodePath = pathParts[pathParts.length - 1] === entry.name ? pathParts.slice(0, -1) : pathParts;
        let currentNode = worldviewEntriesNode;
        nodePath.forEach((part, index) => {
          currentNode = getChild(
            currentNode,
            `wv-path:${nodePath.slice(0, index + 1).join('>')}`,
            part,
            0,
            nodePath.slice(0, index + 1),
          );
        });
        currentNode.entries.push(entry);
        return;
      }

      if (entry.type === 'outline') {
        getNovelOutlineNode(entry).entries.push(entry);
        return;
      }

      if (entry.type === 'prose') {
        const outlineNode = getNovelOutlineNode(entry);
        const chapterNode = getChild(outlineNode, 'chapters', '章节', 0);
        chapterNode.entries.push(entry);
        return;
      }

      const fallbackNode = entry.outline_id
        ? getChild(getNovelOutlineNode(entry), 'related-materials', '关联资料', 1)
        : worldviewEntriesNode;
      fallbackNode.entries.push(entry);
    });

    const pruneEmptyNodes = (node: LoreTreeNode): boolean => {
      Object.entries(node.children).forEach(([key, child]) => {
        if (!pruneEmptyNodes(child)) {
          delete node.children[key];
        }
      });
      return node.path === 'root' || node.entries.length > 0 || Object.keys(node.children).length > 0;
    };

    pruneEmptyNodes(root);

    return root;
  }, [entries, selectedWV, selectedWorldId, worldviews, worlds]);

  const activeWorldview = useMemo(
    () => (worldviews.length === 1 ? worldviews[0] : null),
    [worldviews],
  );

  const getEntryPathParts = (entry: LoreEntry) => {
    const rawParts = normalizeHierarchyPath(entry.path || entry.category || '');
    if (rawParts.length === 0) {
      return entry.name ? [entry.name] : [];
    }
    return rawParts[rawParts.length - 1] === entry.name ? rawParts : [...rawParts, entry.name].filter(Boolean);
  };

  const openWorldviewWorkflow = (
    action: 'create' | 'update',
    entry?: LoreEntry,
    options?: { parentPath?: string[]; message?: string; currentPath?: string; },
  ) => {
    if (!selectedWorldId || !selectedWV) return;
    const params = new URLSearchParams({
      action,
      world_id: selectedWorldId,
      worldview_id: selectedWV,
    });
    if (entry) {
      params.set('id', entry.id);
      params.set('name', entry.name);
      params.set('summary', entry.content || '');
      params.set('path', entry.path || entry.category || '');
    }
    if (options?.parentPath) {
      params.set('parent_path', options.parentPath.join(' > '));
    }
    if (options?.currentPath) {
      params.set('path', options.currentPath);
    }
    if (options?.message) {
      params.set('message', options.message);
    }
    navigate(`/workflow/worldview?${params.toString()}`);
  };

  const openEntryEditor = (entry: LoreEntry) => {
    openWorldviewWorkflow('update', entry);
  };

  const createChildEntry = (parentPath: string[], currentPathLabel: string) => {
    const message = parentPath.length > 0
      ? `在 ${parentPath.join(' > ')} 下新增世界观设定`
      : '新增顶层世界观设定';
    openWorldviewWorkflow('create', undefined, {
      parentPath,
      message,
      currentPath: currentPathLabel,
    });
  };

  const createSiblingEntry = (entry: LoreEntry) => {
    const pathParts = getEntryPathParts(entry);
    createChildEntry(pathParts.slice(0, -1), entry.path || entry.category || pathParts.slice(0, -1).join(' > '));
  };

  const createChildUnderEntry = (entry: LoreEntry) => {
    createChildEntry(getEntryPathParts(entry), entry.path || entry.category || entry.name);
  };

  const toggleTreeNode = (path: string) => {
    setExpandedNodes((current) => ({ ...current, [path]: current[path] === false ? true : false }));
  };

  const isNodeExpanded = (path: string) => expandedNodes[path] !== false;

  const renderTreeNode = (node: LoreTreeNode, depth = 0): React.ReactNode => {
    const children = Object.values(node.children).sort((a, b) => a.order - b.order || a.name.localeCompare(b.name));
    const hasContent = children.length > 0 || node.entries.length > 0;
    const expanded = isNodeExpanded(node.path);

    if (!hasContent) return null;

    return (
      <Box key={node.path}>
        {node.path !== 'root' && (
          <ListItem disablePadding sx={{ pl: depth * 2 }}>
            <ListItemButton onClick={() => toggleTreeNode(node.path)} sx={{ borderRadius: '10px' }}>
              <ListItemIcon sx={{ minWidth: 32, color: 'primary.main' }}>
                {expanded ? <ExpandMoreIcon fontSize="small" /> : <ChevronRightIcon fontSize="small" />}
              </ListItemIcon>
              <ListItemText
                primary={node.name}
                secondary={`${node.entries.length} 条直接设定 / ${children.length} 个子类`}
                primaryTypographyProps={{ variant: 'body2', fontWeight: 800 }}
                secondaryTypographyProps={{ variant: 'caption', sx: { opacity: 0.55 } }}
              />
              {node.hierarchyPath !== undefined && (
                <Tooltip title={node.hierarchyPath.length > 0 ? `在“${node.name}”下新增子设定` : '新增顶层设定'}>
                  <IconButton
                    size="small"
                    onClick={(event) => {
                      event.stopPropagation();
                      createChildEntry(node.hierarchyPath || [], (node.hierarchyPath || []).join(' > '));
                    }}
                    sx={{ color: 'rgba(255,255,255,0.45)' }}
                  >
                    <AddChildIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
            </ListItemButton>
          </ListItem>
        )}

        <Collapse in={node.path === 'root' || expanded} timeout="auto" unmountOnExit>
          {children.map((child) => renderTreeNode(child, node.path === 'root' ? 0 : depth + 1))}
          {node.entries
            .slice()
            .sort((a, b) => a.name.localeCompare(b.name))
            .map((entry) => (
              <ListItem key={entry.id} disablePadding sx={{ pl: (node.path === 'root' ? 0 : depth + 1) * 2 }}>
                <ListItemButton onClick={() => openEntryEditor(entry)} sx={{ borderRadius: '10px', alignItems: 'flex-start' }}>
                  <ListItemIcon sx={{ minWidth: 36, pt: 0.5 }}>
                    <LoreIcon sx={{ fontSize: 18, color: 'rgba(255,255,255,0.55)' }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                        <Typography variant="body2" sx={{ fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {entry.name}
                        </Typography>
                        <Chip label={entry.type} size="small" sx={{ height: 18, fontSize: '0.62rem' }} />
                      </Box>
                    }
                    secondary={entry.content || '无内容'}
                    secondaryTypographyProps={{
                      variant: 'caption',
                      sx: {
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        opacity: 0.65,
                      },
                    }}
                  />
                  <Tooltip title="新增同级设定">
                    <IconButton
                      size="small"
                      onClick={(event) => {
                        event.stopPropagation();
                        createSiblingEntry(entry);
                      }}
                      sx={{ color: 'rgba(255,255,255,0.3)', '&:hover': { color: 'primary.light' } }}
                    >
                      <AddSiblingIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="新增子设定">
                    <IconButton
                      size="small"
                      onClick={(event) => {
                        event.stopPropagation();
                        createChildUnderEntry(entry);
                      }}
                      sx={{ color: 'rgba(255,255,255,0.3)', '&:hover': { color: 'primary.light' } }}
                    >
                      <AddChildIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                  <IconButton
                    size="small"
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDeleteEntry(entry.id);
                    }}
                    sx={{ color: 'rgba(255,255,255,0.25)', '&:hover': { color: 'error.main' } }}
                  >
                    <DeleteIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </ListItemButton>
              </ListItem>
            ))}
        </Collapse>
      </Box>
    );
  };

  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 120px)', gap: 3 }}>
      
      {/* Sidebar: current world's unique worldview set */}
      <Paper className="glass-panel" sx={{ width: 280, p: 2, borderRadius: '20px', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, px: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 800, color: 'primary.main' }}>当前世界设定集</Typography>
          <Chip label="一世界一库" size="small" color="primary" variant="outlined" />
        </Box>
        
        <List sx={{ flexGrow: 1, overflowY: 'auto', px: 0 }}>
          {!selectedWorldId ? (
            <Alert severity="warning" sx={{ borderRadius: '12px' }}>必须选择世界</Alert>
          ) : worldviews.length > 1 ? (
            <Alert severity="error" sx={{ borderRadius: '12px' }}>
              当前世界存在多个设定集，已违反业务规则，请先清理脏数据。
            </Alert>
          ) : worldviews.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 6, opacity: 0.45 }}>
              <WorldviewIcon sx={{ fontSize: 44, mb: 1 }} />
              <Typography variant="body2">当前世界缺少自动创建的世界观设定库</Typography>
            </Box>
          ) : activeWorldview ? (
            <Paper variant="outlined" sx={{ p: 2, borderRadius: '14px', bgcolor: 'rgba(255,255,255,0.02)' }}>
              <Stack spacing={1.25}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <WorldviewIcon sx={{ color: 'primary.main' }} />
                  <Typography variant="body2" sx={{ fontWeight: 800 }}>
                    {getWorldviewDisplayName(activeWorldview)}
                  </Typography>
                </Box>
                <Typography variant="caption" sx={{ opacity: 0.72, wordBreak: 'break-word' }}>
                  库 ID：{activeWorldview.worldview_id}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.72 }}>
                  {activeWorldview.summary || '该世界的所有世界观设定会在右侧按分页显示。'}
                </Typography>
                <Chip label="仅此唯一设定集" size="small" color="primary" sx={{ alignSelf: 'flex-start' }} />
              </Stack>
            </Paper>
          ) : null}
        </List>
      </Paper>

      {/* Main Area: Lore Grid */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {error && <Alert severity="error">{error}</Alert>}
        
        {/* Header Controls */}
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.md,.markdown,.xml,.opml"
            style={{ display: 'none' }}
            onChange={handleImportFileChange}
          />
          <FormControl size="small" required sx={{ minWidth: 280 }}>
            <InputLabel id="lore-world-select-label">世界</InputLabel>
            <Select
              labelId="lore-world-select-label"
              label="世界"
              value={selectedWorldId}
              onChange={(event) => {
                setSelectedWorldId(event.target.value);
                setSelectedWV(null);
                setEntries([]);
                setExpandedNodes({});
              }}
            >
              {worlds.map((world) => (
                <MenuItem key={world.world_id} value={world.world_id}>
                  {world.name} ({world.world_id})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            placeholder="搜索设定实体 (支持语义与关键字)..."
            variant="outlined"
            size="small"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon sx={{ color: 'rgba(255,255,255,0.3)' }} />
                </InputAdornment>
              ),
              sx: { 
                borderRadius: '12px', 
                bgcolor: 'rgba(255,255,255,0.03)',
                '& fieldset': { borderColor: 'rgba(255,255,255,0.1)' }
              }
            }}
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            disabled={!selectedWorldId || !selectedWV || worldviews.length !== 1}
            onClick={() => openWorldviewWorkflow('create')}
            sx={{ 
              borderRadius: '12px', 
              whiteSpace: 'nowrap',
              background: 'linear-gradient(90deg, #00bcd4, #9c27b0)',
              px: 3
            }}
          >
            新增设定
          </Button>
          <Button
            variant="outlined"
            startIcon={<UploadFileIcon />}
            disabled={!selectedWorldId || !selectedWV || worldviews.length !== 1 || importing}
            onClick={openImportPicker}
            sx={{ borderRadius: '12px', whiteSpace: 'nowrap' }}
          >
            {importing ? '导入中...' : '导入设定'}
          </Button>
          <IconButton
            disabled={!selectedWorldId || !selectedWV || worldviews.length !== 1}
            onClick={() => selectedWorldId && selectedWV && fetchEntries(selectedWorldId, selectedWV, searchQuery, entryPage, entryPageSize)}
            sx={{ color: 'rgba(255,255,255,0.5)' }}
          >
            <RefreshIcon />
          </IconButton>
          <ToggleButtonGroup
            size="small"
            exclusive
            value={viewMode}
            onChange={(_, value) => {
              if (value === 'tree' || value === 'table') setViewMode(value);
            }}
            sx={{
              bgcolor: 'rgba(255,255,255,0.03)',
              borderRadius: '12px',
              '& .MuiToggleButton-root': {
                color: 'rgba(255,255,255,0.6)',
                borderColor: 'rgba(255,255,255,0.12)',
                px: 1.5,
              },
              '& .Mui-selected': {
                color: 'primary.main',
                bgcolor: 'rgba(0, 188, 212, 0.12) !important',
              }
            }}
          >
            <ToggleButton value="tree" aria-label="树视图">
              <TreeIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="table" aria-label="表格视图">
              <TableIcon fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* Entries: Tree/Table */}
        <Box sx={{ flexGrow: 1, overflowY: 'auto', pr: 1 }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
              <CircularProgress color="primary" />
            </Box>
          ) : !selectedWorldId ? (
            <Alert severity="warning">必须选择世界后才能查看世界观内容。</Alert>
          ) : worldviews.length > 1 ? (
            <Alert severity="error">当前世界存在多个世界观设定集，/worldviews 只能管理唯一设定集下的设定条目，请先清理脏数据。</Alert>
          ) : !selectedWV ? (
            <Alert severity="warning">当前世界缺少唯一世界观设定库，无法加载设定条目。</Alert>
          ) : viewMode === 'tree' ? (
            <Paper className="glass-panel" sx={{ borderRadius: '18px', overflow: 'hidden' }}>
              <Box sx={{ px: 2, py: 1.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TreeIcon sx={{ color: 'primary.main' }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>层级树视图</Typography>
                </Box>
                <Chip label={`第 ${entryPage} 页 / 当前 ${entries.length} 条`} size="small" />
              </Box>
              <List dense sx={{ p: 1.5 }}>
                {entries.length > 0 ? renderTreeNode(loreTree) : (
                  <Box sx={{ textAlign: 'center', py: 10, opacity: 0.35 }}>
                    <LoreIcon sx={{ fontSize: 60, mb: 2 }} />
                    <Typography>该世界观下暂无设定条目</Typography>
                  </Box>
                )}
              </List>
            </Paper>
          ) : (
            <TableContainer component={Paper} className="glass-panel" sx={{ borderRadius: '18px' }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 800, bgcolor: 'rgba(22,22,37,0.95)' }}>名称</TableCell>
                    <TableCell sx={{ fontWeight: 800, bgcolor: 'rgba(22,22,37,0.95)' }}>类型</TableCell>
                    <TableCell sx={{ fontWeight: 800, bgcolor: 'rgba(22,22,37,0.95)' }}>分类</TableCell>
                    <TableCell sx={{ fontWeight: 800, bgcolor: 'rgba(22,22,37,0.95)', minWidth: 360 }}>内容</TableCell>
                    <TableCell sx={{ fontWeight: 800, bgcolor: 'rgba(22,22,37,0.95)' }}>时间</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 800, bgcolor: 'rgba(22,22,37,0.95)' }}>操作</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {entries.map((entry) => (
                    <TableRow key={entry.id} hover sx={{ cursor: 'pointer' }} onClick={() => openEntryEditor(entry)}>
                      <TableCell sx={{ fontWeight: 700, maxWidth: 220 }}>
                        <Typography variant="body2" noWrap>{entry.name}</Typography>
                        <Typography variant="caption" sx={{ opacity: 0.45 }} noWrap>{entry.id}</Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={entry.type} size="small" sx={{ fontSize: '0.68rem' }} />
                      </TableCell>
                      <TableCell sx={{ maxWidth: 240 }}>
                        <Typography variant="body2" noWrap>{entry.category}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{
                            color: 'text.secondary',
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                            lineHeight: 1.45,
                          }}
                        >
                          {entry.content || '无内容'}
                        </Typography>
                      </TableCell>
                      <TableCell sx={{ opacity: 0.65, whiteSpace: 'nowrap' }}>
                        {entry.timestamp?.split('T')[0] || 'Unknown'}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="编辑">
                          <IconButton
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              openEntryEditor(entry);
                            }}
                          >
                            <EditIcon sx={{ fontSize: 17 }} />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="新增同级设定">
                          <IconButton
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              createSiblingEntry(entry);
                            }}
                          >
                            <AddSiblingIcon sx={{ fontSize: 17 }} />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="新增子设定">
                          <IconButton
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              createChildUnderEntry(entry);
                            }}
                          >
                            <AddChildIcon sx={{ fontSize: 17 }} />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="删除">
                          <IconButton
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              handleDeleteEntry(entry.id);
                            }}
                            sx={{ color: 'rgba(255,255,255,0.35)', '&:hover': { color: 'error.main' } }}
                          >
                            <DeleteIcon sx={{ fontSize: 17 }} />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                  {entries.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} align="center" sx={{ py: 10, opacity: 0.45 }}>
                        该世界观下暂无设定条目
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
        <Paper className="glass-panel" sx={{ borderRadius: '18px', p: 1.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
            <Typography variant="caption" sx={{ opacity: 0.72 }}>
              /worldviews 会分页读取当前世界唯一设定库下的世界观设定；支持 md、json、xml、opml 导入并保留层级路径；树节点支持“在此节点下新增”，设定条目支持“新增同级/新增子设定”，当前第 {entryPage} 页，每页 {entryPageSize} 条。
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <FormControl size="small" sx={{ minWidth: 110 }}>
                <InputLabel id="lore-page-size-label">每页条数</InputLabel>
                <Select
                  labelId="lore-page-size-label"
                  label="每页条数"
                  value={String(entryPageSize)}
                  onChange={(event) => {
                    setEntryPage(1);
                    setEntryPageSize(Number(event.target.value));
                  }}
                >
                  {[10, 20, 50].map((size) => (
                    <MenuItem key={size} value={String(size)}>
                      {size}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Button variant="outlined" disabled={entryPage <= 1 || loading} onClick={() => setEntryPage((current) => Math.max(1, current - 1))}>
                上一页
              </Button>
              <Button variant="outlined" disabled={!hasMoreEntries || loading || !selectedWV || worldviews.length !== 1} onClick={() => setEntryPage((current) => current + 1)}>
                下一页
              </Button>
            </Box>
          </Box>
        </Paper>
      </Box>

      {/* --- Modals --- */}

    </Box>
  );
};
