/**
 * Government Application Header
 * Agent: Frontend_MaterialUI
 */

import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Button,
  Avatar,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  AccountCircle,
  Notifications,
  Settings,
  ExitToApp
} from '@mui/icons-material';

interface HeaderProps {
  onMenuClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick }) => {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleProfileClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <AppBar position=\"fixed\" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        <IconButton
          color=\"inherit\"
          aria-label=\"open drawer\"
          edge=\"start\"
          onClick={onMenuClick}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>
        
        <Typography variant=\"h6\" component=\"div\" sx={{ flexGrow: 1 }}>
          Government Plain Language Compliance System
        </Typography>
        
        <Chip
          label=\"DEMO\"
          size=\"small\"
          color=\"secondary\"
          sx={{ mr: 2 }}
        />
        
        <IconButton color=\"inherit\" sx={{ mr: 1 }}>
          <Notifications />
        </IconButton>
        
        <IconButton
          color=\"inherit\"
          onClick={handleProfileClick}
          aria-controls=\"profile-menu\"
          aria-haspopup=\"true\"
        >
          <Avatar sx={{ bgcolor: 'secondary.main', width: 32, height: 32 }}>
            JD
          </Avatar>
        </IconButton>
        
        <Menu
          id=\"profile-menu\"
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleClose}
          keepMounted
        >
          <MenuItem onClick={handleClose}>
            <AccountCircle sx={{ mr: 1 }} />
            Profile
          </MenuItem>
          <MenuItem onClick={handleClose}>
            <Settings sx={{ mr: 1 }} />
            Settings
          </MenuItem>
          <MenuItem onClick={handleClose}>
            <ExitToApp sx={{ mr: 1 }} />
            Sign Out
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
};

export default Header;