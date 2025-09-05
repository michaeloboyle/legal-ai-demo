import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';

const AgencyDashboard: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Agency Dashboard  
      </Typography>
      <Card>
        <CardContent>
          <Typography>
            Agency-specific compliance dashboard will be implemented here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default AgencyDashboard;