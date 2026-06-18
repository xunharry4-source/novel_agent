import React, { useState } from 'react';
import {
  Box, Drawer, AppBar, Toolbar, List, Typography,
  IconButton, ListItem, ListItemButton, ListItemIcon, ListItemText,
  Container, CssBaseline, ThemeProvider
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Book as BookIcon,
  LibraryBooks as LoreIcon,
  AccountTree as AccountTreeIcon,
  AccountCircle as AccountCircleIcon,
  Login as LoginIcon
} from '@mui/icons-material';
import { useNavigate, Outlet, useLocation } from 'react-router-dom';
import { theme } from '../theme';

const drawerWidth = 260;

export const MainLayout: React.FC = () => {
  const [open, setOpen] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { text: '世界层级', icon: <DashboardIcon />, path: '/worlds' },
    { text: '世界观设定', icon: <LoreIcon />, path: '/worldviews' },
    { text: '星际图谱', icon: <AccountTreeIcon />, path: '/visualizer' },
    { text: '设定知识库', icon: <BookIcon />, path: '/lore' },
    { text: '小说项目管理', icon: <DashboardIcon />, path: '/novels' },
    { text: '用户信息', icon: <AccountCircleIcon />, path: '/me' },
    { text: '登录/注册', icon: <LoginIcon />, path: '/login' },
  ] as { text: string; icon: JSX.Element; path: string; color?: string }[];

  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', bgcolor: 'background.default', minHeight: '100vh' }}>
        <CssBaseline />
        <AppBar
          position="fixed"
          sx={{
            zIndex: (theme) => theme.zIndex.drawer + 1,
            bgcolor: 'rgba(10, 10, 15, 0.8)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
            boxShadow: 'none'
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              onClick={handleDrawerToggle}
              edge="start"
              sx={{ marginRight: 5 }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{
              fontWeight: 800,
              background: 'linear-gradient(90deg, #00bcd4, #9c27b0)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              万象星际 Novel Agent
            </Typography>
          </Toolbar>
        </AppBar>
        <Drawer
          variant="permanent"
          open={open}
          sx={{
            width: open ? drawerWidth : 70,
            transition: theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
            '& .MuiDrawer-paper': {
              width: open ? drawerWidth : 70,
              bgcolor: 'background.paper',
              borderRight: '1px solid rgba(255, 255, 255, 0.08)',
              overflowX: 'hidden',
              transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto', mt: 2 }}>
            <List>
              {menuItems.map((item) => (
                <ListItem key={item.text} disablePadding sx={{ display: 'block' }}>
                  {(() => {
                    const isActive = location.pathname.startsWith(item.path);
                    return (
                      <ListItemButton
                        onClick={() => navigate(item.path)}
                        selected={isActive}
                        sx={{
                          minHeight: 48,
                          justifyContent: open ? 'initial' : 'center',
                          px: 2.5,
                          mx: 1,
                          my: 0.5,
                          borderRadius: 2,
                          bgcolor: isActive ? 'rgba(0, 188, 212, 0.15)' : 'transparent',
                          borderLeft: isActive ? '3px solid #00bcd4' : '3px solid transparent',
                          '&.Mui-selected': {
                            bgcolor: 'rgba(0, 188, 212, 0.2)',
                          },
                          '&:hover': {
                            bgcolor: 'rgba(0, 188, 212, 0.1)',
                          }
                        }}
                      >
                        <ListItemIcon
                          sx={{
                            minWidth: 0,
                            mr: open ? 3 : 'auto',
                            justifyContent: 'center',
                            color: isActive ? '#00bcd4' : (item.color || 'primary.main'),
                          }}
                        >
                          {item.icon}
                        </ListItemIcon>
                        <ListItemText
                          primary={item.text}
                          sx={{
                            opacity: open ? 1 : 0,
                            fontWeight: isActive ? 800 : 600,
                            color: isActive ? '#fff' : 'inherit'
                          }}
                        />
                      </ListItemButton>
                    );
                  })()}
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            mt: 8,
            height: 'calc(100vh - 64px)',
            overflowY: 'auto',
            overflowX: 'hidden',
          }}
        >
          <Container maxWidth="xl">
            <Outlet />
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
};
