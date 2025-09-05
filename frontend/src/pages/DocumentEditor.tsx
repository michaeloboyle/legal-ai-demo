/**
 * Government Document Editor with Real-time Compliance
 * Agent: Frontend_MaterialUI & MikeBostock_Visualization
 * GitHub Issues: #8, #9
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Snackbar
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Description as DocumentIcon,
  Assessment as MetricsIcon,
  Security as ComplianceIcon,
  Speed as PerformanceIcon,
  Save as SaveIcon,
  Download as DownloadIcon,
  Refresh as OptimizeIcon
} from '@mui/icons-material';

// Types
interface ComplianceMetrics {
  originalGradeLevel: number;
  optimizedGradeLevel: number;
  gradeLevelImprovement: number;
  legalAccuracyScore: number;
  plainWritingActCompliance: number;
  semanticSimilarity: number;
  processingTimeMs: number;
}

interface OptimizationResponse {
  originalText: string;
  optimizedText: string;
  metrics: ComplianceMetrics;
  complianceStatus: 'COMPLIANT' | 'NEEDS_IMPROVEMENT' | 'NON_COMPLIANT';
  suggestions: string[];
  timestamp: string;
}

// Compliance visualization component
const ComplianceGauge: React.FC<{ score: number; label: string }> = ({ score, label }) => {
  const getColor = (score: number) => {
    if (score >= 0.9) return '#4caf50';
    if (score >= 0.7) return '#ff9800';
    return '#f44336';
  };

  return (
    <Box sx={{ textAlign: 'center', p: 2 }}>
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          variant=\"determinate\"
          value={score * 100}
          size={80}
          thickness={6}
          sx={{ color: getColor(score) }}
        />
        <Box
          sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: 'absolute',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography variant=\"h6\" component=\"div\" color=\"text.secondary\">
            {Math.round(score * 100)}%
          </Typography>
        </Box>
      </Box>
      <Typography variant=\"body2\" sx={{ mt: 1 }}>
        {label}
      </Typography>
    </Box>
  );
};

const DocumentEditor: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [originalText, setOriginalText] = useState('');
  const [optimizedText, setOptimizedText] = useState('');
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [metrics, setMetrics] = useState<ComplianceMetrics | null>(null);
  const [complianceStatus, setComplianceStatus] = useState<string>('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({ open: false, message: '', severity: 'info' });

  // Sample government text for demonstration
  const sampleText = `The aforementioned regulatory provisions shall be implemented pursuant to the requirements set forth in 29 CFR 1926.95, notwithstanding any conflicting interpretations heretofore promulgated by the administrative authority. All covered entities must demonstrate compliance with the specified standards through documentation that substantiates adherence to the prescribed methodologies, provided that such documentation is maintained in accordance with the record-keeping requirements established under the applicable regulatory framework.`;

  // Load sample text on component mount
  useEffect(() => {
    setOriginalText(sampleText);
  }, []);

  // API call to optimize document
  const optimizeDocument = useCallback(async () => {
    if (!originalText.trim()) {
      setNotification({
        open: true,
        message: 'Please enter text to optimize',
        severity: 'warning'
      });
      return;
    }

    setIsOptimizing(true);
    
    try {
      const response = await fetch('/api/v1/optimize-government-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document: originalText,
          target_grade_level: 10,
          preserve_legal_accuracy: true,
          agency: 'DEMO'
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to optimize document');
      }

      const result: OptimizationResponse = await response.json();
      
      setOptimizedText(result.optimizedText);
      setMetrics(result.metrics);
      setComplianceStatus(result.complianceStatus);
      setSuggestions(result.suggestions);
      
      setNotification({
        open: true,
        message: `Document optimized successfully! Grade level improved from ${result.metrics.originalGradeLevel.toFixed(1)} to ${result.metrics.optimizedGradeLevel.toFixed(1)}`,
        severity: 'success'
      });

      // Switch to results tab
      setActiveTab(1);
      
    } catch (error) {
      console.error('Error optimizing document:', error);
      setNotification({
        open: true,
        message: 'Failed to optimize document. Please try again.',
        severity: 'error'
      });
    } finally {
      setIsOptimizing(false);
    }
  }, [originalText]);

  // Real-time grade level calculation (simplified)
  const calculateGradeLevel = (text: string): number => {
    const words = text.split(/\\s+/).length;
    const sentences = text.split(/[.!?]+/).length;
    const syllables = text.split(/\\s+/).reduce((acc, word) => {
      return acc + Math.max(1, word.length / 3); // Simplified syllable count
    }, 0);

    if (sentences === 0 || words === 0) return 0;

    // Simplified Flesch-Kincaid formula
    const gradeLevel = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59;
    return Math.max(0, gradeLevel);
  };

  const currentGradeLevel = calculateGradeLevel(originalText);

  return (
    <Box>
      <Typography variant=\"h4\" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <DocumentIcon sx={{ mr: 2 }} />
        Government Document Optimizer
      </Typography>
      
      <Typography variant=\"body1\" color=\"text.secondary\" gutterBottom>
        Transform complex government documents into plain language while preserving legal accuracy.
      </Typography>

      <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} sx={{ mb: 3 }}>
        <Tab label=\"Document Input\" icon={<DocumentIcon />} />
        <Tab label=\"Optimized Results\" icon={<MetricsIcon />} />
        <Tab label=\"Compliance Analysis\" icon={<ComplianceIcon />} />
      </Tabs>

      {/* Tab Panel 1: Document Input */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant=\"h6\" gutterBottom>
                  Original Government Document
                </Typography>
                <TextField
                  multiline
                  rows={12}
                  fullWidth
                  value={originalText}
                  onChange={(e) => setOriginalText(e.target.value)}
                  placeholder=\"Paste your government document here...\"
                  variant=\"outlined\"
                  sx={{ mb: 2 }}
                />
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                  <Button
                    variant=\"contained\"
                    color=\"primary\"
                    size=\"large\"
                    startIcon={isOptimizing ? <CircularProgress size={20} color=\"inherit\" /> : <OptimizeIcon />}
                    onClick={optimizeDocument}
                    disabled={isOptimizing || !originalText.trim()}
                  >
                    {isOptimizing ? 'Optimizing...' : 'Optimize for Plain Language'}
                  </Button>
                  <Button
                    variant=\"outlined\"
                    startIcon={<SaveIcon />}
                    disabled={!originalText.trim()}
                  >
                    Save Draft
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant=\"h6\" gutterBottom>
                  Current Document Analysis
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant=\"body2\" color=\"text.secondary\" gutterBottom>
                    Reading Grade Level
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant=\"h4\" color={currentGradeLevel > 12 ? 'error' : 'text.primary'}>
                      {currentGradeLevel.toFixed(1)}
                    </Typography>
                    <Chip 
                      label={currentGradeLevel > 12 ? 'Too Complex' : currentGradeLevel > 8 ? 'Good' : 'Excellent'}
                      color={currentGradeLevel > 12 ? 'error' : currentGradeLevel > 8 ? 'warning' : 'success'}
                      size=\"small\"
                      sx={{ ml: 1 }}
                    />
                  </Box>
                  <LinearProgress 
                    variant=\"determinate\" 
                    value={Math.min(100, (currentGradeLevel / 16) * 100)}
                    color={currentGradeLevel > 12 ? 'error' : 'primary'}
                  />
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant=\"body2\" color=\"text.secondary\" gutterBottom>
                    Word Count: {originalText.split(/\\s+/).filter(word => word.length > 0).length}
                  </Typography>
                  <Typography variant=\"body2\" color=\"text.secondary\" gutterBottom>
                    Character Count: {originalText.length}
                  </Typography>
                  <Typography variant=\"body2\" color=\"text.secondary\">
                    Estimated Reading Time: {Math.ceil(originalText.split(/\\s+/).length / 200)} min
                  </Typography>
                </Box>

                {currentGradeLevel > 10 && (
                  <Alert severity=\"warning\" sx={{ mt: 2 }}>
                    This document exceeds the Plain Writing Act recommendation of 10th grade level.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Tab Panel 2: Optimized Results */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant=\"h6\" gutterBottom color=\"text.secondary\">
                  Original Text
                </Typography>
                <Paper 
                  elevation={0} 
                  sx={{ 
                    p: 2, 
                    backgroundColor: 'grey.50', 
                    maxHeight: 400, 
                    overflow: 'auto' 
                  }}
                >
                  <Typography variant=\"body2\">
                    {originalText || 'No original text available'}
                  </Typography>
                </Paper>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant=\"h6\" color=\"primary\">
                    Optimized Text
                  </Typography>
                  {metrics && (
                    <Chip
                      label={`${metrics.gradeLevelImprovement.toFixed(1)} grade levels improved`}
                      color=\"success\"
                      size=\"small\"
                    />
                  )}
                </Box>
                <Paper 
                  elevation={0} 
                  sx={{ 
                    p: 2, 
                    backgroundColor: 'primary.50', 
                    maxHeight: 400, 
                    overflow: 'auto',
                    border: '1px solid',
                    borderColor: 'primary.main'
                  }}
                >
                  <Typography variant=\"body2\">
                    {optimizedText || 'Click \"Optimize for Plain Language\" to see results'}
                  </Typography>
                </Paper>
                
                {optimizedText && (
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Button
                      variant=\"outlined\"
                      size=\"small\"
                      startIcon={<SaveIcon />}
                    >
                      Save Optimized
                    </Button>
                    <Button
                      variant=\"outlined\"
                      size=\"small\"
                      startIcon={<DownloadIcon />}
                    >
                      Download
                    </Button>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Performance Metrics */}
          {metrics && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant=\"h6\" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12} sm={6} md={3}>
                      <ComplianceGauge 
                        score={metrics.legalAccuracyScore} 
                        label=\"Legal Accuracy\" 
                      />
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                      <ComplianceGauge 
                        score={metrics.plainWritingActCompliance} 
                        label=\"PWA Compliance\" 
                      />
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                      <ComplianceGauge 
                        score={metrics.semanticSimilarity} 
                        label=\"Semantic Similarity\" 
                      />
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ textAlign: 'center', p: 2 }}>
                        <Typography variant=\"h4\" color=\"primary\">
                          {metrics.processingTimeMs.toFixed(0)}ms
                        </Typography>
                        <Typography variant=\"body2\">
                          Processing Time
                        </Typography>
                        <Chip
                          label={metrics.processingTimeMs < 2000 ? 'Fast' : 'Slow'}
                          color={metrics.processingTimeMs < 2000 ? 'success' : 'warning'}
                          size=\"small\"
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}

      {/* Tab Panel 3: Compliance Analysis */}
      {activeTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            {suggestions.length > 0 ? (
              <Box>
                <Typography variant=\"h6\" gutterBottom>
                  Compliance Recommendations
                </Typography>
                {suggestions.map((suggestion, index) => (
                  <Accordion key={index}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>Recommendation {index + 1}</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography>{suggestion}</Typography>
                    </AccordionDetails>
                  </Accordion>
                ))}
              </Box>
            ) : (
              <Card>
                <CardContent sx={{ textAlign: 'center', py: 6 }}>
                  <ComplianceIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant=\"h6\" color=\"text.secondary\">
                    Optimize a document to see compliance analysis
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant=\"h6\" gutterBottom>
                  Plain Writing Act Requirements
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant=\"body2\" gutterBottom>
                    ✓ Use common, everyday words
                  </Typography>
                  <Typography variant=\"body2\" gutterBottom>
                    ✓ Keep sentences to 20 words or less
                  </Typography>
                  <Typography variant=\"body2\" gutterBottom>
                    ✓ Use active voice
                  </Typography>
                  <Typography variant=\"body2\" gutterBottom>
                    ✓ Use \"you\" to address readers
                  </Typography>
                  <Typography variant=\"body2\" gutterBottom>
                    ✓ Keep paragraphs short
                  </Typography>
                </Box>
                
                {complianceStatus && (
                  <Alert 
                    severity={complianceStatus === 'COMPLIANT' ? 'success' : 
                             complianceStatus === 'NEEDS_IMPROVEMENT' ? 'warning' : 'error'}
                  >
                    Status: {complianceStatus.replace('_', ' ')}
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification({ ...notification, open: false })}
      >
        <Alert 
          onClose={() => setNotification({ ...notification, open: false })} 
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DocumentEditor;