import React, { useCallback } from 'react';
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim"; 
import { loadPolygonMaskPlugin } from "tsparticles-plugin-polygon-mask";

const ParticleBackground = ({ enabled = true }) => {
    const particlesInit = useCallback(async engine => {
        await loadSlim(engine);
        await loadPolygonMaskPlugin(engine);
    }, []);

    const particlesLoaded = useCallback(async container => {
        // You can do something here if needed
    }, []);
    const [visible, setVisible] = React.useState(true);

    React.useEffect(() => {
      const handler = () => setVisible(v => !v);
      window.addEventListener('toggleParticlesInternal', handler);
      return () => window.removeEventListener('toggleParticlesInternal', handler);
    }, []);

    const options = {
        background: {
            color: {
                value: "transparent",
            },
        },
        fpsLimit: 60,
        interactivity: {
            events: {
                onHover: {
                    enable: true,
                    mode: "bubble",
                },
                resize: true,
            },
            modes: {
                bubble: {
                    distance: 40,
                    duration: 1,
                    opacity: 1,
                    size: 3,
                },
            },
        },
        particles: {
            color: {
                // default to solid black particles for better contrast
                value: "#7e1313ff",
            },
            links: {
                enable: false,
            },
            collisions: {
                enable: false,
            },
            move: {
                direction: "top",
                enable: true,
                outModes: {
                    default: "out",
                },
                random: false,
                speed: 0.5,
                straight: false,
            },
            number: {
                density: {
                    enable: true,
                    area: 80000,
                },
                // reduce count so black particles 'pop'
                value: 80,
            },
            opacity: {
                value: 0.9,
                animation: {
                    enable: true,
                    speed: 0.9,
                    minimumValue: 0.7,
                },
            },
            shape: {
                type: "circle",
            },
            size: {
                value: { min: 1, max: 3 },
            },
        },
        detectRetina: true,
        polygon: {
            enable: true,
            type: 'inline',
            inline: {
                policy: 'path'
            },
            data: {
                path: 'M18 12a1 1 0 0 0-1-1h-2a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1z',
                size: {
                    width: 100,
                    height: 100
                }
            },
            scale: 5,
            draw: {
                enable: true,
                stroke: {
                    color: 'hsl(var(--foreground))',
                    width: 0.5,
                    opacity: 0.1
                }
            }
        }
    };

    if (!visible) return null;
    return (
        <div className="absolute inset-0 z-0 pointer-events-none">
            <Particles
                id="tsparticles"
                init={particlesInit}
                loaded={particlesLoaded}
                options={options}
                height="100vh"
                width="100vw"
            />
        </div>
    );
};

export default ParticleBackground;