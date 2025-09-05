import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const Settings: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Card>
        <CardContent>
          <Typography>
            System settings and configuration will be implemented here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Settings;