"use client";

import { useEffect, useRef, useState } from "react";

type Particle = {
  id: number;
  x: number;
  y: number;
  char: "0" | "1";
  size: number;
  driftX: number;
  driftY: number;
  duration: number;
  peakOpacity: number;
};

const SPAWN_THROTTLE_MS = 90;
const PARTICLE_LIFETIME_MS = 1600;
const MAX_PARTICLES = 28;

export function CursorTrail() {
  const [particles, setParticles] = useState<Particle[]>([]);
  const idRef = useRef(0);
  const lastSpawnRef = useRef(0);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (motionQuery.matches) return;

    const isCoarsePointer = window.matchMedia("(pointer: coarse)").matches;
    if (isCoarsePointer) return;

    function onMove(e: MouseEvent) {
      const now = performance.now();
      if (now - lastSpawnRef.current < SPAWN_THROTTLE_MS) return;
      lastSpawnRef.current = now;

      const id = idRef.current++;
      const driftX = (Math.random() - 0.5) * 30;
      const driftY = 18 + Math.random() * 28;
      const newParticle: Particle = {
        id,
        x: e.clientX,
        y: e.clientY,
        char: Math.random() > 0.5 ? "1" : "0",
        size: 11 + Math.random() * 9,
        driftX,
        driftY,
        duration: 1.1 + Math.random() * 0.7,
        peakOpacity: 0.32 + Math.random() * 0.28,
      };

      setParticles((prev) => {
        const next = prev.length >= MAX_PARTICLES ? prev.slice(1) : prev;
        return [...next, newParticle];
      });

      window.setTimeout(() => {
        setParticles((prev) => prev.filter((p) => p.id !== id));
      }, PARTICLE_LIFETIME_MS);
    }

    window.addEventListener("mousemove", onMove, { passive: true });
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-[60] overflow-hidden"
    >
      {particles.map((p) => (
        <span
          key={p.id}
          className="cursor-bit absolute font-mono leading-none select-none"
          style={
            {
              left: `${p.x}px`,
              top: `${p.y}px`,
              fontSize: `${p.size}px`,
              animationDuration: `${p.duration}s`,
              "--bit-drift-x": `${p.driftX}px`,
              "--bit-drift-y": `${p.driftY}px`,
              "--bit-opacity": p.peakOpacity,
            } as React.CSSProperties
          }
        >
          {p.char}
        </span>
      ))}
    </div>
  );
}
