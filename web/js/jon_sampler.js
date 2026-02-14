import { app } from "/scripts/app.js";

app.registerExtension({
    name: "JosephOddNodes.JonSampler",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const image_nodes = ["JonZImageSampler", "JonQwen2511Sampler", "JonFlux2Klein9bSampler"]
        const video_nodes = ["JonWan22Sampler", "JonLTX2Sampler"]
        const targetNodes = [...image_nodes, ...video_nodes];


        if (targetNodes.includes(nodeData.name)) {
            const resizeNodeKeepingWidth = (node) => {
                const currentWidth = node.size[0];
                const minSize = node.computeSize();
                node.setSize([Math.max(currentWidth, minSize[0]), minSize[1]]);
            };

            // UI Element Creators
            const createSeparator = () => ({
                type: "separator", name: "separator",
                draw: (ctx, node, w, y) => {
                    ctx.strokeStyle = "#444"; ctx.beginPath(); ctx.moveTo(0, y+5); ctx.lineTo(w, y+5); ctx.stroke();
                },
                computeSize: (w) => [w, 10]
            });

            const createLabel = (text) => ({
                type: "label", name: "label_" + text,
                draw: (ctx, node, w, y) => {
                    ctx.fillStyle = "#aaa"; ctx.font = "bold 12px Arial"; ctx.fillText(text, 10, y + 15);
                },
                computeSize: (w) => [w, 20]
            });


            const toggleWidget = (node, name, enable) => {
                const w = node.widgets.find(w => w.name === name);
                if (w)
                    w.disabled = !enable;
            };

            const getVal = (node, name) => {
                const w = node.widgets.find(w => w.name === name);
                return w ? w.value : false;
            };

            const refreshVisibility = (node) => {
                // Model
                // LTX upscale_enabled ! upscale_model false
                // ZIMAGE -> ! img2img -> disable denoise and denoise1
            };


            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;



                // all image nodes
                if (image_nodes.includes(nodeData.name)) {
                    const img2imgIdx = this.widgets.findIndex(w => w.name === "img2img");
                    if(img2imgIdx > -1){
                        this.widgets.splice(img2imgIdx, 0, createLabel("Save Settiings"), createSeparator());
                    }else{
                        const save_imageIdx = this.widgets.findIndex(w => w.name === "save_image");
                        this.widgets.splice(save_imageIdx, 0, createLabel("Save Settings"), createSeparator());
                    }
                }

                // all video nodes
                if (video_nodes.includes(nodeData.name)) {
                    const img2imgIdx = this.widgets.findIndex(w => w.name === "img2vid");
                    if(img2imgIdx > -1) {
                        const img2vidIdx = this.widgets.findIndex(w => w.name === "img2vid");
                        this.widgets.splice(img2vidIdx, 0, createLabel("Save Settings"), createSeparator());
                    } else {
                        const save_imageIdx = this.widgets.findIndex(w => w.name === "save_video");
                        this.widgets.splice(save_imageIdx, 0, createLabel("Save Settings"), createSeparator());
                    }


                    const total_framesIdx = this.widgets.findIndex(w => w.name === "total_frames");
                    this.widgets.splice(total_framesIdx, 0, createLabel("Video Settings"), createSeparator());
                }

                // Globals
                const modeOptsIdx = this.widgets.findIndex(w => w.name === "seed");
                this.widgets.splice(modeOptsIdx, 0, createLabel("Global Seed"), createSeparator());

                const widthIdx = this.widgets.findIndex(w => w.name === "width");
                this.widgets.splice(widthIdx, 0, createLabel("Geometry"), createSeparator());

                const positiveIdx = this.widgets.findIndex(w => w.name === "positive");
                this.widgets.splice(positiveIdx, 0, createLabel("Prompt"), createSeparator());


                // per node case

                // ZImage Turbo
                if (nodeData.name === "JonZImageSampler") {
                    const denoiseIdx = this.widgets.findIndex(w => w.name === "denoise");
                    this.widgets.splice(denoiseIdx, 0, createLabel("Other Settiings"), createSeparator());
                }


                // // WAN 2.2 14B
                // if (nodeData.name === "JonWan22Sampler") {
                //     this.widgets.splice(0, 0, createLabel("Model Configuration"), createSeparator());
                //     const fileStartIdx = this.widgets.findIndex(w => w.name === "ckpt_name");
                //     if (fileStartIdx > -1)
                //         this.widgets.splice(fileStartIdx, 0, createLabel("File Selection"), createSeparator());
                // }
                // LTX2
                if (nodeData.name === "JonLTX2Sampler") {
                    const fileStartIdx = this.widgets.findIndex(w => w.name === "ckpt_name");
                    if (fileStartIdx > -1)
                        this.widgets.splice(fileStartIdx, 0, createLabel("File Selection"), createSeparator());

                    // const triggers = ["upscale_enabled"];
                    // triggers.forEach(t => {
                    //     const w = this.widgets.find(w => w.name === t);
                    //     if (w)
                    //         w.callback = () => { refreshVisibility(this); };
                    // });
                    //
                    // setTimeout(() => { refreshVisibility(this); }, 50);
                }

                return r;
            };

            //  onConfigure
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(w) {
                if (onConfigure)
                    onConfigure.apply(this, arguments);

                refreshVisibility(this);
                resizeNodeKeepingWidth(this);
            };

            nodeType.prototype.onExecuted = function (message) {
                if (video_nodes.includes(nodeData.name)) {
                    if (message?.videos) {
                        this.imgs = message.videos.map(v => {
                            const img = new Image();
                            img.src = `/view?filename=${v.filename}&type=${v.type}&subfolder=${v.subfolder}`;
                            return img;
                        });
                    }
                } else {
                    if (message?.images) {
                        this.imgs = message.images.map((img) => {
                            const img_obj = new Image();
                            img_obj.src = `/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder)}&t=${+new Date()}`;
                            return img_obj;
                        });
                    }
                }
                this.setDirtyCanvas(true, true);
            };

        }
    }
});
