import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
    startOnLoad: true,
    theme: 'dark',
    securityLevel: 'loose',
    fontFamily: 'monospace',
});

const MermaidDiagram = ({ chart, id }) => {
    const containerRef = useRef(null);
    const [svg, setSvg] = useState('');

    useEffect(() => {
        const renderDiagram = async () => {
            if (!containerRef.current || !chart) return;
            try {
                const { svg } = await mermaid.render(`mermaid-${id}`, chart);
                setSvg(svg);
            } catch (error) {
                console.error('Mermaid render error:', error);
                setSvg(`<div class="text-danger p-4">Error rendering diagram</div>`);
            }
        };

        renderDiagram();
    }, [chart, id]);

    return (
        <div 
            ref={containerRef}
            className="mermaid-container w-full h-full overflow-auto p-4 flex justify-center"
            dangerouslySetInnerHTML={{ __html: svg }}
        />
    );
};

export default MermaidDiagram;
