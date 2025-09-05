"""
Government Plain Language Compliance API
Agent: MLEngineering_Lead
GitHub Issue: #7
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
from datetime import datetime

from ..services.knowledge_distillation import KnowledgeDistillationService
from ..services.compliance_scorer import ComplianceScorer
from ..services.evaluation import EvaluationService
from ..utils.auth import authenticate_government_user
from ..utils.cache import RedisCache
from ..utils.monitoring import track_metrics, log_compliance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Government Plain Language Compliance API",
    description="AI-powered Plain Writing Act compliance system using knowledge distillation",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration for government domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*.gov",
        "http://localhost:3000"  # Development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize services
kd_service = KnowledgeDistillationService()
compliance_scorer = ComplianceScorer()
evaluation_service = EvaluationService()
cache = RedisCache()

# Request/Response Models
class DocumentOptimizationRequest(BaseModel):
    """Request model for document optimization"""
    document: str = Field(..., description="Original government document text")
    target_grade_level: int = Field(10, ge=6, le=12, description="Target reading grade level")
    preserve_legal_accuracy: bool = Field(True, description="Preserve legal accuracy above threshold")
    agency: Optional[str] = Field(None, description="Government agency identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document": "The aforementioned regulatory provisions shall be implemented...",
                "target_grade_level": 10,
                "preserve_legal_accuracy": True,
                "agency": "DOL"
            }
        }

class ComplianceMetrics(BaseModel):
    """Compliance metrics for optimized document"""
    original_grade_level: float
    optimized_grade_level: float
    grade_level_improvement: float
    legal_accuracy_score: float
    plain_writing_act_compliance: float
    semantic_similarity: float
    processing_time_ms: float

class DocumentOptimizationResponse(BaseModel):
    """Response model for document optimization"""
    original_text: str
    optimized_text: str
    metrics: ComplianceMetrics
    compliance_status: str
    suggestions: List[str]
    timestamp: datetime

class BatchDocument(BaseModel):
    """Model for batch document processing"""
    id: str
    document: str
    agency: str

class BatchOptimizationRequest(BaseModel):
    """Request model for batch optimization"""
    documents: List[BatchDocument]
    target_grade_level: int = 10
    priority: str = Field("normal", pattern="^(low|normal|high|critical)$")

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, str]
    timestamp: datetime

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Government Plain Language Compliance API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "health": "/api/health"
    }

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "knowledge_distillation": kd_service.health_check(),
            "compliance_scoring": compliance_scorer.health_check(),
            "cache": cache.health_check()
        },
        "timestamp": datetime.utcnow()
    }
    
    # Check if any service is unhealthy
    if any(status != "healthy" for status in health_status["services"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.post("/api/v1/optimize-government-text", response_model=DocumentOptimizationResponse)
@track_metrics
async def optimize_document(
    request: DocumentOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(authenticate_government_user)
):
    """
    Optimize government document for plain language compliance
    
    This endpoint:
    1. Applies knowledge distillation to simplify text
    2. Preserves legal accuracy above 90% threshold
    3. Achieves target reading grade level
    4. Returns compliance metrics and suggestions
    """
    start_time = time.time()
    
    try:
        # Check cache for previously optimized document
        cache_key = f"opt:{hash(request.document)}:{request.target_grade_level}"
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for document optimization - Agency: {request.agency}")
            return cached_result
        
        # Apply knowledge distillation
        optimized_text = await kd_service.optimize_document(
            document=request.document,
            target_grade_level=request.target_grade_level,
            preserve_legal_accuracy=request.preserve_legal_accuracy
        )
        
        # Calculate compliance metrics
        metrics = await compliance_scorer.calculate_metrics(
            original=request.document,
            optimized=optimized_text
        )
        
        # Evaluate against Plain Writing Act requirements
        compliance_evaluation = await evaluation_service.evaluate_compliance(
            optimized_text,
            metrics
        )
        
        # Prepare response
        processing_time = (time.time() - start_time) * 1000
        
        response = DocumentOptimizationResponse(
            original_text=request.document,
            optimized_text=optimized_text,
            metrics=ComplianceMetrics(
                original_grade_level=metrics["original_grade_level"],
                optimized_grade_level=metrics["optimized_grade_level"],
                grade_level_improvement=metrics["grade_level_improvement"],
                legal_accuracy_score=metrics["legal_accuracy_score"],
                plain_writing_act_compliance=metrics["compliance_score"],
                semantic_similarity=metrics["semantic_similarity"],
                processing_time_ms=processing_time
            ),
            compliance_status=compliance_evaluation["status"],
            suggestions=compliance_evaluation["suggestions"],
            timestamp=datetime.utcnow()
        )
        
        # Cache result
        await cache.set(cache_key, response, ttl=3600)
        
        # Log compliance metrics in background
        background_tasks.add_task(
            log_compliance,
            user_id=current_user["id"],
            agency=request.agency,
            metrics=metrics,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/compliance-score/{document_id}")
async def get_compliance_score(
    document_id: str,
    current_user: Dict = Depends(authenticate_government_user)
):
    """
    Get compliance score for a previously optimized document
    """
    try:
        # Retrieve from cache or database
        score = await compliance_scorer.get_score(document_id)
        
        if not score:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document_id": document_id,
            "compliance_score": score["compliance_score"],
            "grade_level": score["grade_level"],
            "metrics": score["detailed_metrics"],
            "timestamp": score["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving compliance score: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch-optimize")
async def batch_optimize_documents(
    request: BatchOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(authenticate_government_user)
):
    """
    Batch optimization for multiple government documents
    
    Processes documents asynchronously and returns batch ID for tracking
    """
    try:
        # Create batch job
        batch_id = f"batch_{datetime.utcnow().timestamp()}"
        
        # Queue documents for processing
        background_tasks.add_task(
            process_batch_optimization,
            batch_id=batch_id,
            documents=request.documents,
            target_grade_level=request.target_grade_level,
            user_id=current_user["id"]
        )
        
        return {
            "batch_id": batch_id,
            "status": "processing",
            "document_count": len(request.documents),
            "estimated_completion_time": len(request.documents) * 2,  # seconds
            "tracking_url": f"/api/v1/batch-status/{batch_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating batch job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/batch-status/{batch_id}")
async def get_batch_status(
    batch_id: str,
    current_user: Dict = Depends(authenticate_government_user)
):
    """Get status of batch optimization job"""
    try:
        status = await cache.get(f"batch_status:{batch_id}")
        
        if not status:
            raise HTTPException(status_code=404, detail="Batch job not found")
        
        return status
        
    except Exception as e:
        logger.error(f"Error retrieving batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agencies/metrics")
async def get_agency_metrics(
    agency: Optional[str] = None,
    current_user: Dict = Depends(authenticate_government_user)
):
    """
    Get compliance metrics by agency
    
    Returns aggregated metrics for specified agency or all agencies
    """
    try:
        if agency:
            metrics = await evaluation_service.get_agency_metrics(agency)
        else:
            metrics = await evaluation_service.get_all_agency_metrics()
        
        return {
            "agency": agency or "all",
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving agency metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions

async def process_batch_optimization(
    batch_id: str,
    documents: List[BatchDocument],
    target_grade_level: int,
    user_id: str
):
    """Process batch document optimization"""
    results = []
    
    for i, doc in enumerate(documents):
        try:
            # Optimize each document
            optimized = await kd_service.optimize_document(
                document=doc.document,
                target_grade_level=target_grade_level,
                preserve_legal_accuracy=True
            )
            
            # Calculate metrics
            metrics = await compliance_scorer.calculate_metrics(
                original=doc.document,
                optimized=optimized
            )
            
            results.append({
                "document_id": doc.id,
                "status": "completed",
                "optimized_text": optimized,
                "metrics": metrics
            })
            
            # Update batch status
            await cache.set(
                f"batch_status:{batch_id}",
                {
                    "status": "processing",
                    "progress": f"{i+1}/{len(documents)}",
                    "completed": i + 1,
                    "total": len(documents)
                },
                ttl=3600
            )
            
        except Exception as e:
            results.append({
                "document_id": doc.id,
                "status": "failed",
                "error": str(e)
            })
    
    # Final batch status
    await cache.set(
        f"batch_status:{batch_id}",
        {
            "status": "completed",
            "results": results,
            "completed": len(documents),
            "total": len(documents),
            "completion_time": datetime.utcnow().isoformat()
        },
        ttl=86400  # 24 hours
    )

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)