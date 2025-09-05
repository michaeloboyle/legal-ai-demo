/**
 * Government Compliance Dashboard
 * Agent: MikeBostock_Visualization & Frontend_MaterialUI  
 * GitHub Issues: #8, #9
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Alert
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Assignment as DocumentIcon,
  Speed as SpeedIcon,
  Security as ComplianceIcon,
  Group as AgencyIcon,
  Timeline as TimelineIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon
} from '@mui/icons-material';

// Mock data representing government agency performance
const mockAgencyData = {
  totalDocuments: 15743,
  documentsOptimized: 8924,
  averageGradeLevel: 12.4,
  targetGradeLevel: 10.0,
  complianceRate: 0.73,
  averageProcessingTime: 1.8,
  activeAgencies: 12,
  topPerformingAgencies: [
    { name: 'Department of Veterans Affairs', grade: 'A', compliance: 0.94, documents: 2341 },
    { name: 'Social Security Administration', grade: 'A-', compliance: 0.89, documents: 1876 },
    { name: 'Department of Education', grade: 'B+', compliance: 0.84, documents: 1654 },
    { name: 'Department of Labor', grade: 'B', compliance: 0.78, documents: 1423 },
    { name: 'Department of Health & Human Services', grade: 'C+', compliance: 0.71, documents: 2107 }
  ],
  recentActivity: [
    { agency: 'DOL', action: 'Optimized 45 regulatory documents', time: '2 hours ago', status: 'success' },
    { agency: 'VA', action: 'Completed batch optimization of benefits documents', time: '4 hours ago', status: 'success' },
    { agency: 'ED', action: 'Started plain language review of student aid materials', time: '6 hours ago', status: 'in_progress' },
    { agency: 'HHS', action: 'Failed to optimize 3 Medicare policy documents', time: '8 hours ago', status: 'error' },
    { agency: 'SSA', action: 'Published simplified disability determination guidelines', time: '1 day ago', status: 'success' }
  ],
  monthlyTrends: {
    documentsProcessed: [1200, 1350, 1420, 1680, 1534, 1789, 1923, 2103, 2245, 2187, 2340, 2456],
    averageGradeLevel: [14.2, 13.8, 13.5, 13.1, 12.9, 12.7, 12.5, 12.3, 12.1, 12.0, 11.8, 11.6],
    complianceRate: [0.45, 0.52, 0.58, 0.61, 0.65, 0.68, 0.71, 0.73, 0.75, 0.76, 0.78, 0.80]
  }
};

const StatCard: React.FC<{
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  trend?: number;
}> = ({ title, value, subtitle, icon, color = 'primary', trend }) => {
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography color=\"text.secondary\" gutterBottom variant=\"body2\">
              {title}
            </Typography>
            <Typography variant=\"h4\" component=\"div\" color={`${color}.main`}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant=\"body2\" color=\"text.secondary\">
                {subtitle}
              </Typography>
            )}
            {trend !== undefined && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <TrendingUpIcon 
                  sx={{ 
                    fontSize: 16, 
                    color: trend > 0 ? 'success.main' : 'error.main',
                    transform: trend < 0 ? 'rotate(180deg)' : 'none'
                  }} 
                />
                <Typography variant=\"caption\" color={trend > 0 ? 'success.main' : 'error.main'}>
                  {Math.abs(trend)}% vs last month
                </Typography>
              </Box>
            )}
          </Box>
          <Box sx={{ color: `${color}.main` }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState(mockAgencyData);

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setDashboardData(prev => ({
        ...prev,
        totalDocuments: prev.totalDocuments + Math.floor(Math.random() * 5),
        documentsOptimized: prev.documentsOptimized + Math.floor(Math.random() * 3),
        averageProcessingTime: +(prev.averageProcessingTime + (Math.random() - 0.5) * 0.1).toFixed(1)
      }));
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getGradeColor = (grade: string) => {
    if (grade.startsWith('A')) return 'success';
    if (grade.startsWith('B')) return 'warning';
    return 'error';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckIcon color=\"success\" />;
      case 'in_progress':
        return <WarningIcon color=\"warning\" />;
      case 'error':
        return <ErrorIcon color=\"error\" />;
      default:
        return <DocumentIcon />;
    }
  };

  return (
    <Box>
      <Typography variant=\"h4\" gutterBottom>
        Government Plain Language Dashboard
      </Typography>
      
      <Typography variant=\"body1\" color=\"text.secondary\" gutterBottom sx={{ mb: 4 }}>
        Real-time monitoring of federal agency plain language compliance across all departments.
      </Typography>

      {/* Key Performance Indicators */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title=\"Total Documents\"
            value={dashboardData.totalDocuments.toLocaleString()}
            subtitle=\"Federal documents processed\"
            icon={<DocumentIcon sx={{ fontSize: 40 }} />}
            color=\"primary\"
            trend={12.3}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title=\"Optimization Rate\"
            value={`${Math.round((dashboardData.documentsOptimized / dashboardData.totalDocuments) * 100)}%`}
            subtitle={`${dashboardData.documentsOptimized.toLocaleString()} documents optimized`}
            icon={<SpeedIcon sx={{ fontSize: 40 }} />}
            color=\"success\"
            trend={8.7}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title=\"Average Grade Level\"
            value={dashboardData.averageGradeLevel}
            subtitle={`Target: ${dashboardData.targetGradeLevel} grade level`}
            icon={<TimelineIcon sx={{ fontSize: 40 }} />}
            color={dashboardData.averageGradeLevel > dashboardData.targetGradeLevel + 2 ? 'error' : 'warning'}
            trend={-5.2}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title=\"Compliance Rate\"
            value={`${Math.round(dashboardData.complianceRate * 100)}%`}
            subtitle=\"Plain Writing Act compliance\"
            icon={<ComplianceIcon sx={{ fontSize: 40 }} />}
            color={dashboardData.complianceRate > 0.8 ? 'success' : 'warning'}
            trend={15.4}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Agency Performance Ranking */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant=\"h6\" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <AgencyIcon sx={{ mr: 1 }} />
                Agency Performance Ranking
              </Typography>
              <Typography variant=\"body2\" color=\"text.secondary\" gutterBottom>
                Plain Writing Act compliance grades by federal agency
              </Typography>
              
              <List>
                {dashboardData.topPerformingAgencies.map((agency, index) => (
                  <React.Fragment key={agency.name}>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant=\"body1\">
                              #{index + 1} {agency.name}
                            </Typography>
                            <Chip
                              label={agency.grade}
                              color={getGradeColor(agency.grade)}
                              size=\"small\"
                            />
                          </Box>
                        }
                        secondary={
                          <Box sx={{ mt: 1 }}>
                            <Typography variant=\"caption\" display=\"block\">
                              Compliance: {Math.round(agency.compliance * 100)}% • 
                              Documents: {agency.documents.toLocaleString()}
                            </Typography>
                            <LinearProgress
                              variant=\"determinate\"
                              value={agency.compliance * 100}
                              color={getGradeColor(agency.grade)}
                              sx={{ mt: 1 }}
                            />
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < dashboardData.topPerformingAgencies.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity Feed */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant=\"h6\" gutterBottom>
                Recent Activity
              </Typography>
              <Typography variant=\"body2\" color=\"text.secondary\" gutterBottom>
                Live feed of government document optimization activities
              </Typography>
              
              <List>
                {dashboardData.recentActivity.map((activity, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        {getStatusIcon(activity.status)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant=\"body2\">
                            <strong>{activity.agency}:</strong> {activity.action}
                          </Typography>
                        }
                        secondary={activity.time}
                      />
                    </ListItem>
                    {index < dashboardData.recentActivity.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* System Performance */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant=\"h6\" gutterBottom>
                System Performance
              </Typography>
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant=\"body2\">Average Processing Time</Typography>
                  <Typography variant=\"body2\" color=\"primary\">
                    {dashboardData.averageProcessingTime}s
                  </Typography>
                </Box>
                <LinearProgress 
                  variant=\"determinate\" 
                  value={Math.min(100, (2 / dashboardData.averageProcessingTime) * 100)} 
                  color={dashboardData.averageProcessingTime < 2 ? 'success' : 'warning'}
                />
                <Typography variant=\"caption\" color=\"text.secondary\">
                  Target: <2.0 seconds per document
                </Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant=\"body2\">System Uptime</Typography>
                  <Typography variant=\"body2\" color=\"success.main\">99.9%</Typography>
                </Box>
                <LinearProgress variant=\"determinate\" value={99.9} color=\"success\" />
              </Box>

              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant=\"body2\">Active Agencies</Typography>
                  <Typography variant=\"body2\" color=\"info.main\">
                    {dashboardData.activeAgencies}/15
                  </Typography>
                </Box>
                <LinearProgress 
                  variant=\"determinate\" 
                  value={(dashboardData.activeAgencies / 15) * 100} 
                  color=\"info\"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant=\"h6\" gutterBottom>
                Quick Actions
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Button
                    variant=\"contained\"
                    fullWidth
                    startIcon={<DocumentIcon />}
                    href=\"/editor\"
                  >
                    Optimize New Document
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant=\"outlined\"
                    fullWidth
                    startIcon={<ComplianceIcon />}
                    href=\"/metrics\"
                  >
                    View Metrics
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant=\"outlined\"
                    fullWidth
                    startIcon={<AgencyIcon />}
                    href=\"/agency\"
                  >
                    Agency View
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Alerts */}
      <Box sx={{ mt: 4 }}>
        <Alert severity=\"info\" sx={{ mb: 2 }}>
          <strong>System Update:</strong> Knowledge distillation models updated to improve legal accuracy preservation by 3%.
        </Alert>
        
        <Alert severity=\"warning\">
          <strong>Compliance Alert:</strong> 3 agencies are below 70% Plain Writing Act compliance. Review recommended.
        </Alert>
      </Box>
    </Box>
  );
};

export default Dashboard;