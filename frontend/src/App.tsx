import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './layout/MainLayout';
import { LoreDB } from './pages/LoreDB';
import { WorldviewVisualizer } from './pages/WorldviewVisualizer';
import { WorldHierarchy } from './pages/WorldHierarchy';
import { WorldDetail } from './pages/WorldDetail';
import { NovelManagement } from './pages/NovelManagement';
import { NovelCreate } from './pages/NovelCreate';
import { NovelDetail } from './pages/NovelDetail';
import { NovelOutlineManagement } from './pages/NovelOutlineManagement';
import { NovelChapterManagement } from './pages/NovelChapterManagement';
import { NovelChapterContentManagement } from './pages/NovelChapterContentManagement';
import { Login } from './pages/Login';
import { UserProfile } from './pages/UserProfile';
import { RequireAuth } from './components/RequireAuth';
import {
  ChapterWorkflow,
  HierarchyWorkflow,
  NovelWorkflow,
  OutlineWorkflow,
  WorldWorkflow,
  WorldviewWorkflow,
} from './pages/HierarchyWorkflow';
import { MantineProvider } from '@mantine/core';
import '@mantine/core/styles.css';

function App() {
  return (
    <MantineProvider defaultColorScheme="dark">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Navigate to="/worlds" replace />} />
            <Route path="worlds" element={<WorldHierarchy />} />
            <Route path="worlds/:worldId" element={<WorldDetail />} />
            <Route path="worldviews" element={<LoreDB />} />
            <Route path="lore" element={<LoreDB />} />
            <Route path="visualizer" element={<WorldviewVisualizer />} />
            <Route path="login" element={<Login mode="login" />} />
            <Route path="register" element={<Login mode="register" />} />
            <Route
              path="me"
              element={
                <RequireAuth>
                  <UserProfile />
                </RequireAuth>
              }
            />
            <Route path="novels" element={<NovelManagement />} />
            <Route path="novels/new" element={<NovelCreate />} />
            <Route path="novels/:novelId/outlines" element={<NovelOutlineManagement />} />
            <Route path="novels/:novelId/chapters" element={<NovelChapterManagement />} />
            <Route path="novels/:novelId/chapter-contents" element={<NovelChapterContentManagement />} />
            <Route path="novels/:novelId" element={<NovelDetail />} />
            <Route path="workflow" element={<HierarchyWorkflow />} />
            <Route path="workflow/world" element={<WorldWorkflow />} />
            <Route path="workflow/worldview" element={<WorldviewWorkflow />} />
            <Route path="workflow/novel" element={<NovelWorkflow />} />
            <Route path="workflow/outline" element={<OutlineWorkflow />} />
            <Route path="workflow/chapter" element={<ChapterWorkflow />} />
            <Route path="*" element={<Navigate to="/worlds" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </MantineProvider>
  );
}

export default App;
