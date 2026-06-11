(function () {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fitCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * ratio));
    const height = Math.max(1, Math.floor(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: rect.width, height: rect.height };
  }

  const palette = ["#2f7f76", "#c29224", "#b95c50", "#4764a8", "#6d5a9f"];

  const nodes = [
    { id: "chatter_01", x: 0.24, y: 0.26, r: 15, group: 0 },
    { id: "chatter_02", x: 0.32, y: 0.42, r: 28, group: 1 },
    { id: "chatter_03", x: 0.19, y: 0.62, r: 18, group: 0 },
    { id: "chatter_04", x: 0.43, y: 0.29, r: 13, group: 3 },
    { id: "chatter_05", x: 0.51, y: 0.52, r: 34, group: 1 },
    { id: "chatter_06", x: 0.62, y: 0.38, r: 20, group: 2 },
    { id: "chatter_07", x: 0.73, y: 0.22, r: 14, group: 4 },
    { id: "chatter_08", x: 0.76, y: 0.55, r: 24, group: 2 },
    { id: "chatter_09", x: 0.63, y: 0.72, r: 16, group: 3 },
    { id: "chatter_10", x: 0.38, y: 0.74, r: 21, group: 0 },
    { id: "chatter_11", x: 0.84, y: 0.78, r: 12, group: 4 }
  ];

  const edges = [
    [0, 1, 0.72], [1, 2, 0.5], [1, 4, 0.86], [2, 9, 0.42],
    [3, 4, 0.38], [4, 5, 0.68], [5, 6, 0.36], [5, 7, 0.74],
    [7, 8, 0.48], [8, 9, 0.34], [6, 10, 0.28], [0, 4, 0.31],
    [3, 6, 0.24], [4, 8, 0.29], [7, 10, 0.44]
  ];

  function drawNetwork(canvas, options) {
    if (!canvas) return;
    const { context, width, height } = fitCanvas(canvas);
    context.clearRect(0, 0, width, height);

    const time = options.motion ? performance.now() / 1000 : 0;
    const margin = options.margin || 48;

    context.fillStyle = options.background || "#11191b";
    context.fillRect(0, 0, width, height);

    const points = nodes.map((node, index) => {
      const wobble = options.motion ? Math.sin(time * 0.55 + index) * 5 : 0;
      const drift = options.motion ? Math.cos(time * 0.42 + index * 1.6) * 4 : 0;
      return {
        ...node,
        px: margin + node.x * (width - margin * 2) + wobble,
        py: margin + node.y * (height - margin * 2) + drift,
        rr: node.r * (options.scale || 1)
      };
    });

    edges.forEach(([from, to, strength]) => {
      const a = points[from];
      const b = points[to];
      context.beginPath();
      context.moveTo(a.px, a.py);
      context.lineTo(b.px, b.py);
      context.lineWidth = 1 + strength * 4;
      context.strokeStyle = `rgba(255, 250, 241, ${0.08 + strength * 0.18})`;
      context.stroke();
    });

    points.forEach((node) => {
      const color = palette[node.group % palette.length];
      context.beginPath();
      context.arc(node.px, node.py, node.rr + 8, 0, Math.PI * 2);
      context.fillStyle = `${color}24`;
      context.fill();

      context.beginPath();
      context.arc(node.px, node.py, node.rr, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = "rgba(255, 250, 241, 0.72)";
      context.stroke();

      if (options.labels) {
        context.fillStyle = "rgba(255, 250, 241, 0.86)";
        context.font = "12px ui-sans-serif, system-ui, sans-serif";
        context.fillText(node.id, node.px + node.rr + 8, node.py + 4);
      }
    });
  }

  const hero = document.getElementById("hero-network");
  const cluster = document.getElementById("cluster-map");

  function frame() {
    drawNetwork(hero, { motion: !reduceMotion, margin: 96, scale: 0.9 });
    drawNetwork(cluster, { motion: false, labels: true, margin: 72, scale: 1.05 });
    if (!reduceMotion) {
      window.requestAnimationFrame(frame);
    }
  }

  window.addEventListener("resize", () => {
    drawNetwork(hero, { motion: !reduceMotion, margin: 96, scale: 0.9 });
    drawNetwork(cluster, { motion: false, labels: true, margin: 72, scale: 1.05 });
  });

  frame();
})();
