import { app } from "/scripts/app.js";

app.registerExtension({
    name: "JosephOddNodes.ChannelMixer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "JonChannelMixer") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.setSize([480, 480]);

                // ghost inputs
                this.channelWidgets = [];
                const chNames = ["ch_1", "ch_2", "ch_3", "ch_4", "ch_5", "ch_6", "ch_7", "ch_8"];

                if (this.widgets) {
                    for (const name of chNames) {
                        const w = this.widgets.find(w => w.name === name);
                        if (w) {
                            w.type = "converted-widget";
                            w.computeSize = () => [0, -4];
                            w.draw = () => {};
                            w.outline = false;
                            this.channelWidgets.push(w);
                        }
                    }
                }

                // dynamic update
                this.updateVisibleChannels = () => {
                    const countWidget = this.widgets.find(w => w.name === "channels");
                    if (!countWidget)
                        return;

                    const mixer = this.widgets.find(w => w.name === "mixer_board");
                    if (mixer)
                        mixer.bgCache = null;

                    this.setDirtyCanvas(true, true);
                };

                const countWidget = this.widgets.find(w => w.name === "channels");
                if (countWidget) {
                    const origCallback = countWidget.callback;
                    countWidget.callback = (v) => {
                        this.updateVisibleChannels();
                        if (origCallback)
                            origCallback(v);
                    };
                }

                // mixer widget
                const mixerWidget = {
                    name: "mixer_board",
                    type: "custom_mixer",
                    value: 0,
                    options: {},

                    bgCache: null,
                    lastDims: "",

                    computeSize: function(width) { return [width, 350]; },

                      getLimits: function(node) {
                          const minW = node.widgets.find(w => w.name === "min_val");
                          const maxW = node.widgets.find(w => w.name === "max_val");
                          return {
                              min: minW ? minW.value : 0.0,
                              max: maxW ? maxW.value : 1.0
                          };
                      },

                      getGeometry: function(widgetWidth, availableH, channelCount, y) {
                          const spacing = 1;
                          const slotWidth = (widgetWidth - (spacing * (channelCount - 1))) / channelCount;

                          const btnH = 16;
                          const btnGap = 4;
                          const buttonAreaH = (btnH * 3) + (btnGap * 2) + 10;

                          const headerGap = 10;
                          const footerH = 20;

                          const btnBaseY = y + 5;
                          const trackTop = btnBaseY + buttonAreaH + headerGap;
                          const trackHeight = availableH - (trackTop - y) - footerH;

                          return {
                              slotWidth, spacing,
                              btnH, btnGap,
                              btnBaseY,
                              trackTop, trackHeight, trackBottom: trackTop + trackHeight
                          };
                      },

                      getDynamicColor: function(pct) {
                          const t = Math.max(0, Math.min(1, pct));
                          const colorBlue = [68, 136, 255];
                          const colorYellow = [255, 230, 0];
                          const colorRed = [255, 60, 60];

                          let r, g, b;
                          if (t < 0.5) {
                              const localT = t * 2;
                              r = colorBlue[0] + (colorYellow[0] - colorBlue[0]) * localT;
                              g = colorBlue[1] + (colorYellow[1] - colorBlue[1]) * localT;
                              b = colorBlue[2] + (colorYellow[2] - colorBlue[2]) * localT;
                          } else {
                              const localT = (t - 0.5) * 2;
                              r = colorYellow[0] + (colorRed[0] - colorYellow[0]) * localT;
                              g = colorYellow[1] + (colorRed[1] - colorYellow[1]) * localT;
                              b = colorYellow[2] + (colorRed[2] - colorYellow[2]) * localT;
                          }
                          return `rgb(${Math.floor(r)}, ${Math.floor(g)}, ${Math.floor(b)})`;
                      },

                      // CACHE (Static Backgrounds)
                      ensureBackgroundCache: function(node, widgetWidth, availableH, channelCount) {
                          const dimHash = `${widgetWidth},${availableH},${channelCount}`;
                          if (this.bgCache && this.lastDims === dimHash)
                              return;

                          this.bgCache = document.createElement("canvas");
                          this.bgCache.width = widgetWidth;
                          this.bgCache.height = availableH;
                          const ctx = this.bgCache.getContext("2d");
                          this.lastDims = dimHash;

                          const geo = this.getGeometry(widgetWidth, availableH, channelCount, 0);

                          for (let i = 0; i < channelCount; i++) {
                              const stripX = (i * (geo.slotWidth + geo.spacing));
                              const centerX = stripX + (geo.slotWidth / 2);

                              // Background
                              ctx.fillStyle = "#0a0a0a";
                              ctx.fillRect(stripX, 0, geo.slotWidth, availableH);

                              // Ruler Lines
                              ctx.strokeStyle = "#222";
                              ctx.lineWidth = 1;
                              ctx.beginPath();
                              const steps = 10;
                              for (let j = 0; j <= steps; j++) {
                                  const pct = j / steps;
                                  const lineY = geo.trackBottom - (pct * geo.trackHeight);
                                  ctx.moveTo(stripX + 4, lineY);
                                  ctx.lineTo(stripX + geo.slotWidth - 4, lineY);
                              }
                              ctx.stroke();

                              // Label
                              ctx.textAlign = "center";
                              ctx.font = "10px Arial";
                              ctx.fillStyle = "#666";
                              ctx.fillText(i+1, centerX, geo.trackBottom + 12);

                              // Button Placeholders
                              const btnW = geo.slotWidth - 4;
                              const btnX = stripX + 2;
                              let currBtnY = geo.btnBaseY;
                              for(let b=0; b<3; b++){
                                  ctx.fillStyle = "#1a1a1a";
                                  ctx.fillRect(btnX, currBtnY, btnW, geo.btnH);
                                  currBtnY += geo.btnH + geo.btnGap;
                              }
                          }
                      },

                      draw: function(ctx, node, widgetWidth, y, widgetHeight) {
                          const bottomMargin = 15;
                          const availableH = node.size[1] - y - bottomMargin;
                          if (availableH < 50)
                              return;

                          const countWidget = node.widgets.find(w => w.name === "channels");
                          const channelCount = countWidget ? countWidget.value : 8;

                          this.ensureBackgroundCache(node, widgetWidth, availableH, channelCount);
                          if (this.bgCache)
                              ctx.drawImage(this.bgCache, 0, y);

                          const geo = this.getGeometry(widgetWidth, availableH, channelCount, y);
                          this.hitBox = { ...geo, channelCount };

                          const { min, max } = this.getLimits(node);
                          const range = max - min;

                          if (!node.channelWidgets)
                              return;

                          for (let i = 0; i < channelCount; i++) {
                              const w = node.channelWidgets[i];
                              if(!w)
                                  continue;

                              const val = w.value;
                              const isMuted = w._isMuted === true;

                              const stripX = (i * (geo.slotWidth + geo.spacing));
                              const centerX = stripX + (geo.slotWidth / 2);
                              const btnX = stripX + 2;
                              const btnW = geo.slotWidth - 4;

                              // buttons
                              let currBtnY = geo.btnBaseY;
                              if (isMuted) {
                                  ctx.fillStyle = "#cc2222";
                                  ctx.fillRect(btnX, currBtnY, btnW, geo.btnH);
                              }

                              ctx.fillStyle = isMuted ? "#fff" : "#555";
                              ctx.font = "bold 9px Arial"; ctx.textAlign = "center";
                              ctx.fillText("M", centerX, currBtnY + 11);
                              currBtnY += geo.btnH + geo.btnGap;

                              ctx.fillStyle = "#aaa"; ctx.fillText("0", centerX, currBtnY + 11);
                              currBtnY += geo.btnH + geo.btnGap;
                              ctx.fillStyle = "#aaa"; ctx.fillText("1", centerX, currBtnY + 11);

                              // FADER GEOMETRY
                              const pct = Math.max(0, Math.min(1, (val - min) / range));
                              const thumbY = geo.trackBottom - (pct * geo.trackHeight);

                              const zeroPct = (0 - min) / range;
                              const clampedZeroPct = Math.max(0, Math.min(1, zeroPct));
                              const zeroY = geo.trackBottom - (clampedZeroPct * geo.trackHeight);

                              // DYNAMIC GROOVE
                              ctx.fillStyle = "#111";
                              ctx.fillRect(centerX - 2, geo.trackTop, 4, geo.trackHeight);

                              // Center Line - INACTIVE (Top to Thumb)
                              ctx.fillStyle = "#000";
                              ctx.fillRect(centerX - 1, geo.trackTop, 2, thumbY - geo.trackTop);

                              // Center Line - ACTIVE (Thumb to Bottom)
                              const grooveColor = this.getDynamicColor(pct);
                              ctx.fillStyle = isMuted ? "#333" : grooveColor;
                              ctx.fillRect(centerX - 1, thumbY, 2, geo.trackBottom - thumbY);

                              // Zero Marker
                              ctx.fillStyle = "#555"; ctx.fillRect(centerX - 10, zeroY, 20, 1);

                              // Active Fill Bar (Side Bar)
                              // Matches the groove color but originates from Zero
                              ctx.fillStyle = isMuted ? "#333" : grooveColor;
                              let fillY = Math.min(zeroY, thumbY);
                              let fillH = Math.abs(zeroY - thumbY);
                              if (fillH > 0)
                                  ctx.fillRect(centerX - 2, fillY, 4, fillH);

                              // "realistic" FADER cap
                              const capW = geo.slotWidth * 0.8;
                              const capH = 18;

                              ctx.fillStyle = "rgba(0,0,0,0.5)"; // Shadow
                              ctx.fillRect(centerX - (capW/2) + 2, thumbY - (capH/2) + 2, capW, capH);
                              ctx.fillStyle = "#333"; // Body
                              ctx.fillRect(centerX - (capW/2), thumbY - (capH/2), capW, capH);
                              ctx.fillStyle = "#fff"; // Stripe
                              ctx.fillRect(centerX - (capW/2), thumbY - 1, capW, 2);

                              // show value text
                              if (!isMuted) {
                                  ctx.font = "bold 10px Arial";
                                  ctx.fillStyle = "#fff";
                                  ctx.fillText(Number(val).toFixed(2), centerX, thumbY - 10);
                              }
                          }
                      },

                      mouse: function(event, pos, node) {
                          if (event.buttons === 0 && event.type !== "pointerdown")
                              return;

                          if (!this.hitBox)
                              return;

                          const geo = this.hitBox;

                          const totalSlotW = geo.slotWidth + geo.spacing;
                          let idx = Math.floor(pos[0] / totalSlotW);
                          idx = Math.max(0, Math.min(geo.channelCount - 1, idx));
                          const targetWidget = node.channelWidgets[idx];
                          if (!targetWidget)
                              return;

                          const { min, max } = this.getLimits(node);

                          if (event.type === "pointerdown" && pos[1] >= geo.btnBaseY && pos[1] <= geo.trackTop - 5) {
                              const relativeY = pos[1] - geo.btnBaseY;
                              const btnIndex = Math.floor(relativeY / (geo.btnH + geo.btnGap));

                              if (btnIndex === 0) { // MUTE
                                  if (targetWidget._isMuted) {
                                      targetWidget.value = targetWidget._prevVal !== undefined ? targetWidget._prevVal : min;
                                      targetWidget._isMuted = false;
                                  } else {
                                      targetWidget._prevVal = targetWidget.value;
                                      targetWidget.value = Math.max(min, Math.min(max, 0));
                                      targetWidget._isMuted = true;
                                  }
                              }
                              else if (btnIndex === 1) { // ZERO
                                  targetWidget.value = Math.max(min, Math.min(max, 0));
                                  targetWidget._isMuted = false;
                              }
                              else if (btnIndex === 2) { // PRESET
                                  targetWidget.value = Math.max(min, Math.min(max, 1.0));
                                  targetWidget._isMuted = false;
                              }
                              targetWidget.callback(targetWidget.value);
                              node.setDirtyCanvas(true, true);
                              return true;
                          }

                          if (pos[1] < geo.trackTop)
                              return;

                          let mouseY = Math.max(geo.trackTop, Math.min(geo.trackBottom, pos[1]));
                          const relativeY = mouseY - geo.trackTop;
                          const rawPct = relativeY / geo.trackHeight;
                          const pct = 1.0 - rawPct;

                          let newVal = min + (pct * (max - min));
                          if (Math.abs(newVal) < 0.01)
                              newVal = 0.0;

                          targetWidget._isMuted = false;
                          targetWidget.value = newVal;
                          targetWidget.callback(newVal);
                          node.setDirtyCanvas(true, true);
                      }
                };

                this.addCustomWidget(mixerWidget);
                return r;
            };
        }
    }
});
