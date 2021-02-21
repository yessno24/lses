clear all
close all

load data_ch2007

%Real: 2007, Predict: 2007,2008
points=num(1:12);
points_dim=size(points,2);
points_n=size(points,1);
cluster_n=3;

const=0.75;
a1=const;
a2=const;
a3=const;

mean_mse=[];
mean_rmse=[];
mean_mape=[];
mean_mae=[];
mean_madc=[];
mean_me=[];

mse_all=[];
rmse_all=[];
mape_all=[];
mae_all=[];
madc_all=[];
me_all=[];

pred2020=[];

for expr=1:100
    all_clust_tinggi=[];
    all_clust_sedang=[];
    all_clust_rendah=[];

    %initial cluster center
    rng(expr);
    [max_points idx_max]=max(points);
    [min_points idx_min]=min(points);
    R=max_points-min_points;
    for j=1:points_dim
        clust_cen(:,j)=min_points(j)+R(j)*rand(cluster_n,1);
    end

    err=10;
    thres=0.0001;

    while err>=thres
        %compute u
        u3=[];
        for k=1:cluster_n
            u1=bsxfun(@minus,points,clust_cen(k,:));
            u2=u1.^2;
            u3=[u3 sum(u2,2)+eps];
        end

        u=zeros(points_n,cluster_n);
        [val idx]=min(u3,[],2);
        for i=1:points_n
            u(i,idx(i))=1;
        end

        %update clust_cen
        new_clust_cen=[];
        for k=1:cluster_n
            temp1=zeros(1,points_dim);
            temp2=sum(u,1);
            for i=1:points_n
                temp1=temp1+u(i,k)*points(i,:);
            end
            new_clust_cen=[new_clust_cen;temp1/temp2(k)];
        end

        err1=abs(clust_cen-new_clust_cen);
        err=max(err1(:));
        clust_cen=new_clust_cen;
    end

    clust=[];
    for i=1:points_n
        [val idx]=max(u(i,:));
        clust=[clust;idx];
    end

    clust_max=clust(idx_max);
    clust_min=clust(idx_min);
    new_clust=[];
    for i=1:points_n
        if (clust(i)==clust_max)
            new_clust=[new_clust;1];
        else
            if (clust(i)==clust_min)
                new_clust=[new_clust;3];
            else
                new_clust=[new_clust;2];
            end
        end
    end

    %data prediksi 2007 menggunakan average cluater data real 2007
    clust1=find(new_clust==1);
    avg_clust1=mean(points(clust1));
    clust2=find(new_clust==2);
    avg_clust2=mean(points(clust2));
    clust3=find(new_clust==3);
    avg_clust3=mean(points(clust3));

    clust_tinggi=points(clust1);
    clust_sedang=points(clust2);
    clust_rendah=points(clust3);

    clust_tinggi_norm=(clust_tinggi-min(clust_tinggi))/(max(clust_tinggi)-min(clust_tinggi));
    clust_sedang_norm=(clust_sedang-min(clust_sedang))/(max(clust_sedang)-min(clust_sedang));
    clust_rendah_norm=(clust_rendah-min(clust_rendah))/(max(clust_rendah)-min(clust_rendah));

    if (size(clust_tinggi,1)==1) 
        a1=const;
    else a1=mean(clust_tinggi_norm);
    end

    if (size(clust_sedang,1)==1) 
        a2=const;
    else a2=mean(clust_sedang_norm);
    end

    if (size(clust_rendah,1)==1) 
        a3=const;
    else a3=mean(clust_rendah_norm);
    end

    %prediksi 2008
    F_ok=[];
    F=[];

    for t=1:points_n
        if new_clust(t)==1
            F(t)=a1*points(t)+(1-a1)*avg_clust1;
        else if new_clust(t)==2
                F(t)=a2*points(t)+(1-a2)*avg_clust2;
            else
                F(t)=a3*points(t)+(1-a3)*avg_clust3;
            end
        end
    end
    F=F';
    F_ok=[F_ok F];  

    %----------------------------------------------------------------
    %Real: 2008-2019, Predict: 2009-2020
    mse=[];
    rmse=[];
    mae=[];
	madc=[];
    me=[];
    
    for tahun=1:12
        points=num(12*tahun+1:12*(tahun+1));    
        mse=[mse;immse(points,F)];
        rmse=[rmse;sqrt(immse(points,F))];
        hit_mae=mean(abs(points-F));
        mae=[mae;hit_mae];
        hit_madc=mean(abs(points-mean(F)));
        madc=[madc;hit_madc];
        hit_me=mean(points-F);
        me=[me;hit_me];

        clust1=[];
        clust2=[];
        clust3=[];

        %initial cluster center
        [max_points idx_max]=max(points);
        [min_points idx_min]=min(points);
        clust_cen=[];
        R=max_points-min_points;
        for j=1:points_dim
            clust_cen(:,j)=min_points(j)+R(j)*rand(cluster_n,1);
        end

        err=10;
        thres=0.0001;
        while err>=thres
            %compute u
            u3=[];
            for k=1:cluster_n
                u1=bsxfun(@minus,points,clust_cen(k,:));
                u2=u1.^2;
                u3=[u3 sum(u2,2)+eps];
            end

            u=zeros(points_n,cluster_n);
            [val idx]=min(u3,[],2);
            for i=1:points_n
                u(i,idx(i))=1;
            end

            %update clust_cen
            new_clust_cen=[];
            for k=1:cluster_n
                temp1=zeros(1,points_dim);
                temp2=sum(u,1);
                for i=1:points_n
                    temp1=temp1+u(i,k)*points(i,:);
                end
                new_clust_cen=[new_clust_cen;temp1/temp2(k)];
            end

            err1=abs(clust_cen-new_clust_cen);
            err=max(err1(:));
            clust_cen=new_clust_cen;
        end

        clust=[];
        for i=1:points_n
            [val idx]=max(u(i,:));
            clust=[clust;idx];
        end

        clust_max=clust(idx_max);
        clust_min=clust(idx_min);
        new_clust=[];
        for i=1:points_n
            if (clust(i)==clust_max)
                new_clust=[new_clust;1];
            else
                if (clust(i)==clust_min)
                    new_clust=[new_clust;3];
                else
                    new_clust=[new_clust;2];
                end
            end
        end

        clust1=find(new_clust==1);
        clust2=find(new_clust==2);
        clust3=find(new_clust==3);

        clust_tinggi=points(clust1);
        clust_sedang=points(clust2);
        clust_rendah=points(clust3);

        all_clust_tinggi=[all_clust_tinggi;clust_tinggi];
        all_clust_sedang=[all_clust_sedang;clust_sedang];
        all_clust_rendah=[all_clust_rendah;clust_rendah];

        clust_tinggi_norm=(all_clust_tinggi-min(all_clust_tinggi))/(max(all_clust_tinggi)-min(all_clust_tinggi));
        clust_sedang_norm=(all_clust_sedang-min(all_clust_sedang))/(max(all_clust_sedang)-min(all_clust_sedang));
        clust_rendah_norm=(all_clust_rendah-min(all_clust_rendah))/(max(all_clust_rendah)-min(all_clust_rendah));

        if or((size(all_clust_tinggi,1)==1),all(points(clust1))==0)
            a1=const;
        else a1=mean(clust_tinggi_norm);
        end

        if or((size(all_clust_sedang,1)==1),all(points(clust2))==0)
            a2=const;
        else a2=mean(clust_sedang_norm);
        end

        if or((size(all_clust_rendah,1)==1),all(points(clust3))==0)
            a3=const;
        else a3=mean(clust_rendah_norm);
        end

        Fs=F_ok(:,tahun);
        F=[];
        for t=1:points_n
            if new_clust(t)==1
                F(t)=a1*points(t)+(1-a1)*Fs(t);
            else if new_clust(t)==2
                    F(t)=a2*points(t)+(1-a2)*Fs(t);
                else
                    F(t)=a3*points(t)+(1-a3)*Fs(t);
                end
            end
        end
        F=F';
        F_ok=[F_ok F];   
    end
    F_ok_fix=reshape(F_ok,[12*(tahun+1),1]);
    pred2020=[pred2020 F_ok(:,13)];
    
    mse_all=[mse_all mse(2:12)];
    rmse_all=[rmse_all rmse(2:12)];
    mae_all=[mae_all mae(2:12)];
	madc_all=[madc_all madc(2:12)];
    me_all=[me_all me(2:12)];
end

a=min(mse_all,[],2);
mean(a)

b=min(mae_all,[],2);
mean(b)

c=min(madc_all,[],2);
mean(c)

min_pred2020=min(pred2020,[],2);

x1=1:12*(tahun+1); y1=num(1:12*(tahun+1));
x2=25:12*(tahun+2); y2=F_ok_fix(13:156);

f=figure;
hold all
plot(x1,y1,'-o','color','blue','LineWidth',0.75,'markersize',2,'markerfacecolor','blue');
plot(x2,y2,'-o','color','red','LineWidth',0.75,'markersize',2,'markerfacecolor','red');
legend('actual','forecast')
xlabel('Prediction Year')
ylabel('Rainfall (mm)')
ylim([0 1400]);

xt={'2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'}; 
set(gca,'xtick',1:12:157); 
set(gca,'xticklabel',xt);
set(gca,'FontSize',8);
title('Rainfall forecasting with Learning-based Single Exponential Smoothing');

