import json
import logging
import os
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import NotFound

from config import settings
from schemas.audit import FacilityAuditReport, VisionTelemetry, BoundingBox

logger = logging.getLogger("visionaudit.services.reasoning")


class ReasoningService:
    """Production orchestration layer using modern LangChain LCEL and Gemini."""

    def __init__(
        self,
        api_key: str = "",
        model_name: str = settings.gemini_model,
        temperature: float = settings.gemini_temperature
    ):
        self._custom_api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self._parser = JsonOutputParser(pydantic_object=FacilityAuditReport)
        self._chains: dict[str, any] = {}

    @property
    def api_key(self) -> str:
        """Dynamically resolve Google Gemini API key from settings or os.environ."""
        return self._custom_api_key or settings.google_api_key or os.getenv("GOOGLE_API_KEY", "")

    def _build_lcel_chain(self, model_to_use: str):
        """Builds LCEL pipe operator chain with JsonOutputParser and format instructions."""
        active_key = self.api_key
        if not active_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured. Please supply it via environment variable or .env file."
            )

        # 1. Initialize modern ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_to_use,
            temperature=self.temperature,
            google_api_key=active_key
        )

        # 2. Define structured audit prompt with JSON schema format instructions
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a Senior AI Facility Operations and Safety Compliance Auditor.\n"
                "Your role is to rigorously analyze computer vision detection telemetry "
                "(object frequencies, entity distributions, bounding coordinates, and confidence levels) "
                "from workplace cameras, warehouses, or industrial facility zones.\n\n"
                "Evaluate the environment against 3 core pillars:\n"
                "1. Safety: Physical hazard risks, worker proximity to equipment, safety compliance, emergency egress clear zones.\n"
                "2. Inventory: Asset counts, inventory density, misplaced equipment, stock presence.\n"
                "3. Operational Risk: Flow bottlenecks, congested paths, unmonitored active items, workflow disruptions.\n\n"
                "For every finding, assign a valid category (Safety, Inventory, or Operational Risk), "
                "a calibrated severity level (LOW, MEDIUM, HIGH), an evidence-based observation, "
                "and an actionable recommendation.\n\n"
                "You must strictly output valid JSON adhering to the following schema:\n"
                "{format_instructions}"
            ),
            (
                "human",
                "Conduct a comprehensive facility audit based on the following vision telemetry data:\n\n"
                "Contextual Timestamp: {timestamp}\n"
                "Total Entities Detected: {total_detections}\n"
                "Class Occurrence Summary:\n{class_counts}\n\n"
                "Object Bounding Coordinates & Confidence:\n{detection_details}\n"
            ),
        ]).partial(format_instructions=self._parser.get_format_instructions())

        # 3. Modern LCEL pipe composition: (prompt | llm | parser)
        return prompt | llm | self._parser

    def _format_detection_details(self, detections: List[BoundingBox], max_items: int = 50) -> str:
        """Format detection coordinates and confidence into readable telemetry."""
        if not detections:
            return "No object detections recorded."

        lines = []
        for idx, det in enumerate(detections[:max_items], 1):
            lines.append(
                f"- #{idx} Class: {det.class_name} | Confidence: {det.confidence:.2f} | BBox [x1, y1, x2, y2]: {det.box}"
            )
        if len(detections) > max_items:
            lines.append(f"... and {len(detections) - max_items} more detections.")
        return "\n".join(lines)

    async def audit_telemetry(self, telemetry: VisionTelemetry, timestamp: str) -> FacilityAuditReport:
        """Asynchronously invoke the LCEL chain with automatic fallback if model is deprecated."""
        active_key = self.api_key
        if not active_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")

        formatted_counts = json.dumps(telemetry.class_counts, indent=2)
        formatted_details = self._format_detection_details(telemetry.detections)
        payload = {
            "timestamp": timestamp,
            "total_detections": telemetry.total_count,
            "class_counts": formatted_counts,
            "detection_details": formatted_details,
        }

        # Preferred model and resilient fallback candidates
        candidate_models = [self.model_name]
        if self.model_name != "gemini-3.5-flash":
            candidate_models.append("gemini-3.5-flash")
        if self.model_name != "gemini-flash-latest":
            candidate_models.append("gemini-flash-latest")

        last_error = None
        for model in candidate_models:
            try:
                if model not in self._chains:
                    self._chains[model] = self._build_lcel_chain(model)

                raw_report = await self._chains[model].ainvoke(payload)

                if isinstance(raw_report, FacilityAuditReport):
                    return raw_report
                elif isinstance(raw_report, dict):
                    return FacilityAuditReport.model_validate(raw_report)
                else:
                    return FacilityAuditReport.model_validate_json(str(raw_report))

            except NotFound as not_found_err:
                logger.warning(
                    f"Model '{model}' returned 404 (retired/not found). Attempting next supported model in candidate pool."
                )
                last_error = not_found_err
                continue
            except Exception as exc:
                # If error mentions 404/not found, try next candidate
                if "404" in str(exc) or "not found" in str(exc).lower():
                    logger.warning(f"Model '{model}' not found. Falling back to alternative model.")
                    last_error = exc
                    continue
                # For all other exceptions (e.g. 429), re-raise to trigger backoff
                raise exc

        if last_error:
            raise last_error
        raise RuntimeError("No candidate Gemini models succeeded.")


# Singleton instance
reasoning_service = ReasoningService()
